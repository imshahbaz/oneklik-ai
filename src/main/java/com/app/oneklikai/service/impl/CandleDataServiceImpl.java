package com.app.oneklikai.service.impl;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.CosineTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.tracker.WarmUpTracker;
import ai.djl.translate.NoopTranslator;
import com.app.oneklikai.components.ModelCacheManager;
import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.exceptions.NotFoundException;
import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.dto.response.EvaluationMetrics;
import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.model.dto.response.TrainingReport;
import com.app.oneklikai.model.entity.CandleStick;
import com.app.oneklikai.model.entity.ModelWeightEntity;
import com.app.oneklikai.repo.CandleStickRepo;
import com.app.oneklikai.repo.ModelWeightRepository;
import com.app.oneklikai.service.CandleDataService;
import com.app.oneklikai.util.*;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class CandleDataServiceImpl implements CandleDataService {

    private static final ZoneId MARKET_ZONE = ZoneId.of("Asia/Kolkata");
    private static final int INFERENCE_CHUNK = 256;

    /** A run is only worth keeping in the ensemble if it is close to the best run. */
    private static final double ENSEMBLE_LOSS_TOLERANCE = 1.25;

    private final CandleStickRepo candleStickRepo;
    private final ModelWeightRepository modelWeightRepository;
    private final ModelCacheManager modelCacheManager;

    /** One seeded training attempt and its best-epoch snapshot. */
    private record RunResult(
            long seed,
            int bestEpoch,
            int epochsRun,
            byte[] weights,
            float[][] validationPredictions,
            EvaluationMetrics validationMetrics
    ) {
    }

    @Override
    @SneakyThrows
    public TrainingReport trainModel(String symbol, int sequenceLength, int epochs, float learningRate) {
        String stockSymbol = symbol.toUpperCase();

        List<CandleStick> candles;
        try (var stream = candleStickRepo.findBySymbolOrderByTimestampAsc(symbol)) {
            candles = stream.filter(CandleDataServiceImpl::isTradeable).toList();
        }
        if (candles.isEmpty()) {
            throw new NotFoundException("No candles found for " + stockSymbol);
        }

        DatasetBuilder.SplitDataset split = DatasetBuilder.build(
                candles,
                sequenceLength,
                Constants.DAILY_VALIDATION_FRACTION,
                Constants.DAILY_TEST_FRACTION,
                Constants.DAILY_RECENCY_HALF_LIFE_SAMPLES);

        log.info("Training {} | candles: {} ({} -> {}) | samples train/val/test: {}/{}/{} | features: {}",
                stockSymbol, split.totalCandles(),
                candles.getFirst().getTimestamp(), candles.getLast().getTimestamp(),
                split.train().size(), split.validation().size(), split.test().size(),
                Constants.DAILY_FEATURE_DIM);

        List<RunResult> runs = new ArrayList<>(Constants.DAILY_TRAINING_RESTARTS);
        EvaluationMetrics validationMetrics;
        EvaluationMetrics testMetrics;
        EvaluationMetrics baselineMetrics;
        List<Model> ensembleModels;
        List<RunResult> ensembleRuns;

        try (NDManager manager = NDManager.newBaseManager()) {
            ArrayDataset trainDataset = toDataset(manager, split.train(),
                    Constants.DAILY_SAMPLING_BATCH_SIZE, true);
            ArrayDataset validationDataset = toDataset(manager, split.validation(),
                    Constants.DAILY_EVAL_BATCH_SIZE, false);

            // Independent seeds: RNN training on noisy financial targets is high variance, so the run
            // that happens to land in a good basin is worth searching for rather than hoping for.
            for (int restart = 0; restart < Constants.DAILY_TRAINING_RESTARTS; restart++) {
                long seed = Constants.DAILY_BASE_SEED + restart * 7919L;
                runs.add(trainSingleRun(stockSymbol, seed, restart + 1, sequenceLength, epochs,
                        learningRate, trainDataset, split.train().size(), validationDataset, split.validation()));
            }

            runs.sort(Comparator.comparingDouble(run -> run.validationMetrics().loss()));
            double bestLoss = runs.getFirst().validationMetrics().loss();
            ensembleRuns = runs.stream()
                    .filter(run -> run.validationMetrics().loss() <= bestLoss * ENSEMBLE_LOSS_TOLERANCE)
                    .toList();

            // Averaging the retained runs cancels a good part of the per-seed noise; the ensemble is
            // scored on validation exactly the way it will be used at inference time.
            validationMetrics = computeMetrics(
                    average(ensembleRuns.stream().map(RunResult::validationPredictions).toList()),
                    split.validation().labels());

            ensembleModels = new ArrayList<>(ensembleRuns.size());
            for (int i = 0; i < ensembleRuns.size(); i++) {
                ensembleModels.add(loadModel(stockSymbol + "-" + i, ensembleRuns.get(i).weights()));
            }

            testMetrics = computeMetrics(
                    ensemblePredict(ensembleModels, split.test().inputs()),
                    split.test().labels());
            baselineMetrics = naiveBaseline(split.test().labels());
        }

        RunResult best = runs.getFirst();
        List<byte[]> ensembleWeights = ensembleRuns.stream().map(RunResult::weights).toList();
        int modelSizeKb = ensembleWeights.stream().mapToInt(bytes -> bytes.length).sum() / 1024;

        ModelWeightEntity weightEntity = modelWeightRepository.findBySymbol(stockSymbol)
                .orElseGet(() -> ModelWeightEntity.builder().symbol(stockSymbol).build());

        weightEntity.setWeightBytes(best.weights());
        weightEntity.setEnsembleWeightBytes(ensembleWeights);
        weightEntity.setSequenceLength(sequenceLength);
        weightEntity.setFeatureDim(Constants.DAILY_FEATURE_DIM);
        weightEntity.setTargetDim(Constants.DAILY_TARGET_DIM);
        weightEntity.setEmbedDim(Constants.DAILY_EMBED_DIM);
        weightEntity.setNumLayers(Constants.DAILY_NUM_LAYERS);
        weightEntity.setDropout(Constants.DAILY_DROPOUT);
        weightEntity.setFeatureVersion(FeatureEngineer.FEATURE_VERSION);
        weightEntity.setTrainSamples(split.train().size());
        weightEntity.setValidationLoss(validationMetrics.loss());
        weightEntity.setValidationDirectionalAccuracyPercent(validationMetrics.directionalAccuracyPercent());
        weightEntity.setTestDirectionalAccuracyPercent(testMetrics.directionalAccuracyPercent());
        weightEntity.setUpdatedAt(Instant.now());
        weightEntity.setTimeFrame(TimeFrame.DAILY);
        modelWeightRepository.save(weightEntity);
        modelCacheManager.setModels(stockSymbol, ensembleModels, TimeFrame.DAILY);

        log.info("""
                        Trained {} | ensemble of {} (best seed {} @ epoch {})
                        validation: loss {} | close MAE {}% | dir-acc {}%
                        test:       loss {} | close MAE {}% | dir-acc {}%
                        baseline:   loss {} | close MAE {}% | dir-acc {}%
                        persisted {} KB""",
                stockSymbol, ensembleModels.size(), best.seed(), best.bestEpoch(),
                format(validationMetrics.loss()), format(validationMetrics.closeMaePercent()),
                format(validationMetrics.directionalAccuracyPercent()),
                format(testMetrics.loss()), format(testMetrics.closeMaePercent()),
                format(testMetrics.directionalAccuracyPercent()),
                format(baselineMetrics.loss()), format(baselineMetrics.closeMaePercent()),
                format(baselineMetrics.directionalAccuracyPercent()),
                modelSizeKb);

        return new TrainingReport(
                stockSymbol,
                OffsetDateTime.ofInstant(weightEntity.getUpdatedAt(), MARKET_ZONE),
                split.totalCandles(),
                sequenceLength,
                Constants.DAILY_FEATURE_DIM,
                split.train().size(),
                split.validation().size(),
                split.test().size(),
                runs.size(),
                ensembleModels.size(),
                best.seed(),
                best.bestEpoch(),
                runs.stream().map(RunResult::epochsRun).toList(),
                validationMetrics,
                testMetrics,
                baselineMetrics,
                modelSizeKb);
    }

    /**
     * Trains one seeded model and returns its best-validation snapshot.
     *
     * <p>Everything here exists to stop the model from memorising the training years:
     * warm-up + cosine learning-rate schedule, gradient clipping (an LSTM unrolled over 60 steps
     * explodes without it), decoupled weight decay, and early stopping on a purged, chronologically
     * later validation slice with the best epoch — not the last — restored at the end.
     */
    @SneakyThrows
    private RunResult trainSingleRun(String stockSymbol,
                                     long seed,
                                     int runNumber,
                                     int sequenceLength,
                                     int epochs,
                                     float learningRate,
                                     ArrayDataset trainDataset,
                                     int trainSamples,
                                     ArrayDataset validationDataset,
                                     DatasetBuilder.WindowSet validationSet) {
        Engine.getInstance().setRandomSeed((int) seed);

        int batchesPerEpoch = Math.max(1,
                (int) Math.ceil(trainSamples / (double) Constants.DAILY_SAMPLING_BATCH_SIZE));
        int decaySteps = Math.max(1, batchesPerEpoch * epochs - Constants.DAILY_WARMUP_STEPS);

        Tracker learningRateTracker = WarmUpTracker.builder()
                .setMainTracker(CosineTracker.builder()
                        .setBaseValue(learningRate)
                        .optFinalValue(learningRate * Constants.DAILY_FINAL_LR_FRACTION)
                        .setMaxUpdates(decaySteps)
                        .build())
                .optWarmUpSteps(Constants.DAILY_WARMUP_STEPS)
                .optWarmUpBeginValue(learningRate / 100f)
                .optWarmUpMode(WarmUpTracker.Mode.LINEAR)
                .build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(
                new WeightedHuberLoss("weightedHuber", Constants.DAILY_HUBER_DELTA, FeatureScaler.LOSS_WEIGHTS))
                .optOptimizer(Optimizer.adamW()
                        .optLearningRateTracker(learningRateTracker)
                        .optWeightDecays(Constants.DAILY_WEIGHT_DECAY)
                        .optClipGrad(Constants.DAILY_GRAD_CLIP)
                        .build());

        double bestLoss = Double.MAX_VALUE;
        byte[] bestWeights = null;
        float[][] bestPredictions = null;
        EvaluationMetrics bestMetrics = null;
        int bestEpoch = 0;
        int epochsRun = 0;

        Model model = Model.newInstance(stockSymbol + "-run" + runNumber);
        try {
            model.setBlock(DataTransformerBlock.buildArchitecture(
                    Constants.DAILY_FEATURE_DIM,
                    Constants.DAILY_TARGET_DIM,
                    Constants.DAILY_EMBED_DIM,
                    Constants.DAILY_NUM_LAYERS,
                    Constants.DAILY_DROPOUT));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, sequenceLength, Constants.DAILY_FEATURE_DIM));

                int epochsWithoutImprovement = 0;
                for (int epoch = 1; epoch <= epochs; epoch++) {
                    epochsRun = epoch;
                    double trainingLoss = runTrainingEpoch(trainer, trainDataset);

                    float[][] predictions = predict(trainer, validationDataset, validationSet.size());
                    EvaluationMetrics metrics = computeMetrics(predictions, validationSet.labels());

                    if (metrics.loss() < bestLoss - 1e-7) {
                        bestLoss = metrics.loss();
                        bestEpoch = epoch;
                        bestMetrics = metrics;
                        bestPredictions = predictions;
                        bestWeights = MemoryModelSerializer.serializeToBytes(model);
                        epochsWithoutImprovement = 0;
                    } else {
                        epochsWithoutImprovement++;
                    }

                    if (epoch % 5 == 0 || epoch == epochs || epochsWithoutImprovement == 0) {
                        log.info("Run {} (seed {}) epoch {}/{} - train {} | val {} | val dir-acc {}% | best epoch {}",
                                runNumber, seed, epoch, epochs, format(trainingLoss),
                                format(metrics.loss()), format(metrics.directionalAccuracyPercent()), bestEpoch);
                    }

                    if (epochsWithoutImprovement >= Constants.DAILY_EARLY_STOP_PATIENCE) {
                        log.info("Run {} (seed {}) early stopped at epoch {}; best epoch {} (val loss {})",
                                runNumber, seed, epoch, bestEpoch, format(bestLoss));
                        break;
                    }
                }
            }
        } finally {
            model.close();
        }

        if (bestWeights == null) {
            throw new IllegalStateException("Training produced no usable snapshot for " + stockSymbol);
        }

        return new RunResult(seed, bestEpoch, epochsRun, bestWeights, bestPredictions, bestMetrics);
    }

    /** One pass over the training data; returns the sample-weighted average batch loss. */
    private double runTrainingEpoch(Trainer trainer, ArrayDataset dataset) {
        double weightedLoss = 0.0;
        long samples = 0;

        try {
            for (Batch miniBatch : trainer.iterateDataset(dataset)) {
                try (Batch batch = miniBatch;
                     GradientCollector collector = trainer.newGradientCollector()) {
                    NDList predictions = trainer.forward(batch.getData());
                    NDArray loss = trainer.getLoss().evaluate(batch.getLabels(), predictions);
                    collector.backward(loss);
                    trainer.step();

                    int batchSize = batch.getSize();
                    weightedLoss += loss.getFloat() * batchSize;
                    samples += batchSize;
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException("Training failed!", e);
        }

        return samples == 0 ? Double.NaN : weightedLoss / samples;
    }

    /** Runs the model in inference mode (dropout off) over an ordered dataset. */
    @SneakyThrows
    private static float[][] predict(Trainer trainer, ArrayDataset dataset, int sampleCount) {
        float[][] predictions = new float[sampleCount][];
        int cursor = 0;

        for (Batch miniBatch : trainer.iterateDataset(dataset)) {
            try (Batch batch = miniBatch;
                 NDList output = trainer.evaluate(batch.getData())) {
                float[] flat = output.singletonOrThrow().toFloatArray();
                int batchSize = batch.getSize();
                for (int i = 0; i < batchSize && cursor < sampleCount; i++) {
                    predictions[cursor++] = Arrays.copyOfRange(flat,
                            i * Constants.DAILY_TARGET_DIM, (i + 1) * Constants.DAILY_TARGET_DIM);
                }
            }
        }

        return predictions;
    }

    /** Averages the raw (still-normalised) outputs of every model in the ensemble. */
    @SneakyThrows
    private static float[][] ensemblePredict(List<Model> models, float[][][] inputs) {
        int samples = inputs.length;
        int targetDim = Constants.DAILY_TARGET_DIM;
        float[][] summed = new float[samples][targetDim];

        for (Model model : models) {
            try (NDManager manager = NDManager.newBaseManager();
                 Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
                for (int from = 0; from < samples; from += INFERENCE_CHUNK) {
                    int to = Math.min(samples, from + INFERENCE_CHUNK);
                    try (NDManager chunkManager = manager.newSubManager()) {
                        NDArray tensor = DataTransformerBlock.create3DTensor(
                                chunkManager, Arrays.copyOfRange(inputs, from, to));
                        try (NDList output = predictor.predict(new NDList(tensor))) {
                            float[] flat = output.singletonOrThrow().toFloatArray();
                            for (int i = from; i < to; i++) {
                                for (int f = 0; f < targetDim; f++) {
                                    summed[i][f] += flat[(i - from) * targetDim + f];
                                }
                            }
                        }
                    }
                }
            }
        }

        for (float[] row : summed) {
            for (int f = 0; f < targetDim; f++) {
                row[f] /= models.size();
            }
        }

        return summed;
    }

    private static float[][] average(List<float[][]> predictionSets) {
        int samples = predictionSets.getFirst().length;
        int targetDim = Constants.DAILY_TARGET_DIM;
        float[][] averaged = new float[samples][targetDim];

        for (float[][] predictions : predictionSets) {
            for (int i = 0; i < samples; i++) {
                for (int f = 0; f < targetDim; f++) {
                    averaged[i][f] += predictions[i][f] / predictionSets.size();
                }
            }
        }

        return averaged;
    }

    /**
     * Scores predictions against labels. Mirrors {@link WeightedHuberLoss} for the loss term so early
     * stopping and the reported numbers come from one implementation.
     */
    private static EvaluationMetrics computeMetrics(float[][] predictions, float[][][] labels) {
        int samples = labels.length;
        float delta = Constants.DAILY_HUBER_DELTA;
        double lossSum = 0.0;
        double absoluteError = 0.0;
        double squaredError = 0.0;
        int correctDirection = 0;

        for (int i = 0; i < samples; i++) {
            float[] label = labels[i][0];
            float[] prediction = predictions[i];

            double weighted = 0.0;
            for (int f = 0; f < Constants.DAILY_TARGET_DIM; f++) {
                double scaledError = (label[f] - prediction[f]) / delta;
                weighted += FeatureScaler.LOSS_WEIGHTS[f] * delta * delta
                        * (Math.sqrt(1.0 + scaledError * scaledError) - 1.0);
            }
            lossSum += weighted * label[FeatureScaler.WEIGHT_INDEX] / Constants.DAILY_TARGET_DIM;

            double predictedPercent = FeatureScaler.toPercent(prediction[FeatureScaler.CLOSE_INDEX]);
            double actualPercent = FeatureScaler.toPercent(label[FeatureScaler.CLOSE_INDEX]);
            double error = predictedPercent - actualPercent;
            absoluteError += Math.abs(error);
            squaredError += error * error;

            if ((prediction[FeatureScaler.CLOSE_INDEX] >= 0) == (label[FeatureScaler.CLOSE_INDEX] >= 0)) {
                correctDirection++;
            }
        }

        if (samples == 0) {
            return new EvaluationMetrics(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0);
        }

        return new EvaluationMetrics(
                lossSum / samples,
                absoluteError / samples,
                Math.sqrt(squaredError / samples),
                100.0 * correctDirection / samples,
                samples);
    }

    /**
     * "Tomorrow closes exactly where today closed", with the best constant direction guess. Any model
     * that cannot beat this on the test split is not worth deploying.
     */
    private static EvaluationMetrics naiveBaseline(float[][][] labels) {
        EvaluationMetrics flat = computeMetrics(new float[labels.length][Constants.DAILY_TARGET_DIM], labels);
        double upShare = flat.directionalAccuracyPercent();
        return new EvaluationMetrics(
                flat.loss(),
                flat.closeMaePercent(),
                flat.closeRmsePercent(),
                Math.max(upShare, 100.0 - upShare),
                flat.samples());
    }

    private static ArrayDataset toDataset(NDManager manager,
                                          DatasetBuilder.WindowSet windows,
                                          int batchSize,
                                          boolean shuffle) {
        return new ArrayDataset.Builder()
                .setData(DataTransformerBlock.create3DTensor(manager, windows.inputs()))
                .optLabels(DataTransformerBlock.create3DTensor(manager, windows.labels()))
                .setSampling(batchSize, shuffle)
                .build();
    }

    @SneakyThrows
    private static Model loadModel(String name, byte[] weights) {
        Model model = Model.newInstance(name);
        model.setBlock(DataTransformerBlock.buildArchitecture(
                Constants.DAILY_FEATURE_DIM,
                Constants.DAILY_TARGET_DIM,
                Constants.DAILY_EMBED_DIM,
                Constants.DAILY_NUM_LAYERS,
                Constants.DAILY_DROPOUT));
        MemoryModelSerializer.deserializeFromBytes(model, weights, name);
        return model;
    }

    private static boolean isTradeable(CandleStick candle) {
        return candle.getTimestamp() != null
                && candle.getOpen() > 0
                && candle.getHigh() > 0
                && candle.getLow() > 0
                && candle.getClose() > 0
                && candle.getVolume() >= 0;
    }

    private static String format(double value) {
        return String.format("%.6f", value);
    }

    @Override
    public PredictionResponse predictNextDay(String symbol, int sequenceLength) {
        String stockSymbol = symbol.toUpperCase();
        List<Model> models = modelCacheManager.getModels(stockSymbol, TimeFrame.DAILY)
                .orElseThrow(() -> new NotFoundException("Model not found!"));

        // Indicators need history before the window itself, so fetch the warm-up candles too.
        int lookback = Math.max(Constants.DAILY_INFERENCE_LOOKBACK,
                sequenceLength + FeatureEngineer.WARMUP + 10);
        List<CandleStick> history = candleStickRepo
                .findBySymbolOrderByTimestampDesc(symbol, PageRequest.of(0, lookback))
                .stream()
                .filter(CandleDataServiceImpl::isTradeable)
                .toList()
                .reversed();

        DatasetBuilder.InferenceWindow window = DatasetBuilder.buildLatestWindow(history, sequenceLength);
        float[] prediction = ensemblePredict(models, window.input())[0];

        double anchorClose = window.anchorClose();
        double predictedOpen = FeatureScaler.decodePrice(prediction[0], anchorClose);
        double predictedHigh = FeatureScaler.decodePrice(prediction[1], anchorClose);
        double predictedLow = FeatureScaler.decodePrice(prediction[2], anchorClose);
        double predictedClose = FeatureScaler.decodePrice(prediction[3], anchorClose);
        double predictedVolume = FeatureScaler.decodeVolume(prediction[4], window.anchorVolumeEma());

        // The four price heads are predicted independently, so enforce the candle invariants.
        predictedHigh = Math.max(predictedHigh, Math.max(predictedOpen, predictedClose));
        predictedLow = Math.min(predictedLow, Math.min(predictedOpen, predictedClose));

        double returnPercent = ((predictedClose - anchorClose) / anchorClose) * 100.0;

        return new PredictionResponse(
                stockSymbol,
                OffsetDateTime.ofInstant(window.anchorCandle().getTimestamp(), MARKET_ZONE),
                round(predictedOpen),
                round(predictedHigh),
                round(predictedLow),
                round(predictedClose),
                Math.round(predictedVolume),
                predictedClose >= anchorClose ? "BULLISH" : "BEARISH",
                round(returnPercent));
    }

    private static double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
