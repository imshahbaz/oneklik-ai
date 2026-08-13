package com.app.oneklikai.service.impl;

import ai.djl.Model;
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
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import com.app.oneklikai.components.ModelCacheManager;
import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.exceptions.NotFoundException;
import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.model.entity.CandleStick;
import com.app.oneklikai.model.entity.ModelWeightEntity;
import com.app.oneklikai.repo.CandleStickRepo;
import com.app.oneklikai.repo.ModelWeightRepository;
import com.app.oneklikai.service.CandleDataService;
import com.app.oneklikai.util.DataTransformerBlock;
import com.app.oneklikai.util.DatasetBuilder;
import com.app.oneklikai.util.FeatureScaler;
import com.app.oneklikai.util.MemoryModelSerializer;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class CandleDataServiceImpl implements CandleDataService {

    private final CandleStickRepo candleStickRepo;
    private final ModelWeightRepository modelWeightRepository;
    private final ModelCacheManager modelCacheManager;

    @Override
    @SneakyThrows
    public void trainModel(String symbol, int sequenceLength, int epochs, float learningRate) {
        String stockSymbol = symbol.toUpperCase();
        Model model = Model.newInstance(stockSymbol);
        try (var stream = candleStickRepo.findBySymbolOrderByTimestampAsc(symbol)) {
            DatasetBuilder.TrainingBatch batch = DatasetBuilder.buildSlidingWindows(stream, sequenceLength);
            int numSamples = batch.inputSequences().length;
            try (NDManager manager = NDManager.newBaseManager()) {
                model.setBlock(DataTransformerBlock.buildArchitecture(Constants.DAILY_FEATURE_DIM, Constants.DAILY_EMBED_DIM));
                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                        .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build());

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.initialize(new Shape(1, sequenceLength, Constants.DAILY_FEATURE_DIM));
                    NDArray inputTensor = DataTransformerBlock.create3DTensor(manager, batch.inputSequences());
                    NDArray targetTensor = DataTransformerBlock.create3DTensor(manager, batch.targetCandles());

                    log.info("Starting Training on {} | Samples: {} | Epochs: {}...", stockSymbol, numSamples, epochs);

                    ArrayDataset dataset = new ArrayDataset.Builder()
                            .setData(inputTensor)
                            .optLabels(targetTensor)
                            .setSampling(Constants.DAILY_SAMPLING_BATCH_SIZE, true)
                            .build();

                    for (int epoch = 1; epoch <= epochs; epoch++) {
                        float accumulatedLoss = 0.0f;
                        int batchCount = 0;

                        try {
                            for (Batch miniBatch : trainer.iterateDataset(dataset)) {
                                try (GradientCollector gc = trainer.newGradientCollector()) {
                                    NDArray predictions = trainer.forward(miniBatch.getData()).singletonOrThrow();
                                    NDArray loss = trainer.getLoss().evaluate(miniBatch.getLabels(), new NDList(predictions));
                                    gc.backward(loss);
                                    trainer.step();
                                    accumulatedLoss += loss.getFloat();
                                    batchCount++;
                                }
                                miniBatch.close();
                            }
                        } catch (Exception e) {
                            throw new IllegalStateException("Training failed!",e);
                        }

                        if (epoch % 5 == 0 || epoch == epochs) {
                            float avgLoss = accumulatedLoss / Math.max(1, batchCount);
                            log.info("Epoch {}/{} - Avg Batch Loss: {}", epoch, epochs, String.format("%.6f", avgLoss));
                        }
                    }
                }
            }
        }

        byte[] weightBytes = MemoryModelSerializer.serializeToBytes(model);
        ModelWeightEntity weightEntity = modelWeightRepository.findBySymbol(stockSymbol)
                .orElseGet(() -> ModelWeightEntity.builder().symbol(stockSymbol).build());

        weightEntity.setWeightBytes(weightBytes);
        weightEntity.setSequenceLength(sequenceLength);
        weightEntity.setFeatureDim(Constants.DAILY_FEATURE_DIM);
        weightEntity.setUpdatedAt(Instant.now());
        weightEntity.setTimeFrame(TimeFrame.DAILY);
        modelWeightRepository.save(weightEntity);
        modelCacheManager.setModel(stockSymbol, model, TimeFrame.DAILY);

        log.info("Model successfully trained for {} persisted to MongoDB ({} KB).",
                stockSymbol, (weightBytes.length / 1024));
    }

    @Override
    public PredictionResponse predictNextDay(String symbol, int sequenceLength) {
        var trainedModel = modelCacheManager.getModel(symbol, TimeFrame.DAILY)
                .orElseThrow(() -> new NotFoundException("Model not found!"));

        String stockSymbol = symbol.toUpperCase();

        // 1. Extract the LAST 60 candles to serve as the prompt context
        List<CandleStick> last60 = candleStickRepo.findTop60BySymbolOrderByTimestampDesc(symbol).reversed();
        var latestCandle = last60.getLast();
        double basePrice = last60.getFirst().getOpen();

        // 2. Build single input window array: shape [1, 60, 5]
        float[][][] inputWindow = new float[1][sequenceLength][5];
        for (int i = 0; i < sequenceLength; i++) {
            inputWindow[0][i] = FeatureScaler.normalizeCandle(last60.get(i), basePrice);
        }

        // 3. Run Inference Pass
        float[] predictedNormalized;
        try (NDManager subManager = trainedModel.getNDManager().newSubManager();
             Predictor<NDList, NDList> predictor = trainedModel.newPredictor(new NoopTranslator())) {
            NDArray inputTensor = DataTransformerBlock.create3DTensor(subManager, inputWindow);
            try (NDList outputList = predictor.predict(new NDList(inputTensor))) {
                predictedNormalized = outputList.singletonOrThrow().get(0).get(0).toFloatArray();
            }
        } catch (TranslateException e) {
            throw new IllegalStateException("Processing failed!", e);
        }

        // 4. Denormalize prediction outputs back to actual price scale
        double predOpen = FeatureScaler.denormalizePrice(predictedNormalized[0], basePrice);
        double predHigh = FeatureScaler.denormalizePrice(predictedNormalized[1], basePrice);
        double predLow = FeatureScaler.denormalizePrice(predictedNormalized[2], basePrice);
        double predClose = FeatureScaler.denormalizePrice(predictedNormalized[3], basePrice);
        double predVolume = FeatureScaler.denormalizeVolume(predictedNormalized[4]);

        // 5. Calculate directional metrics relative to today's actual close
        double currentClose = latestCandle.getClose();
        double returnPercent = ((predClose - currentClose) / currentClose) * 100.0;
        String direction = predClose >= currentClose ? "BULLISH" : "BEARISH";

        return new PredictionResponse(
                stockSymbol,
                OffsetDateTime.ofInstant(latestCandle.getTimestamp(), ZoneId.of("Asia/Kolkata")),
                Math.round(predOpen * 100.0) / 100.0,
                Math.round(predHigh * 100.0) / 100.0,
                Math.round(predLow * 100.0) / 100.0,
                Math.round(predClose * 100.0) / 100.0,
                Math.round(predVolume),
                direction,
                Math.round(returnPercent * 100.0) / 100.0
        );
    }
}
