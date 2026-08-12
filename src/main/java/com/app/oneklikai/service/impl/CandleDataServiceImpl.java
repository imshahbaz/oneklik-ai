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
import com.app.oneklikai.constant.Constants;
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
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class CandleDataServiceImpl implements CandleDataService {

    private final CandleStickRepo candleStickRepo;
    private final ModelWeightRepository modelWeightRepository;
    private final MongoTemplate mongoTemplate;

    @EventListener(ApplicationReadyEvent.class)
    public void loadGlobalModelFromDatabase() {
        log.info("Checking for pre-trained global model in database...");
        Optional<ModelWeightEntity> savedWeight = modelWeightRepository.findById(Constants.DAILY_MODEL_NAME);
        var model = Constants.DAILY_MODEL;
        model.setBlock(DataTransformerBlock.buildArchitecture(Constants.DAILY_FEATURE_DIM, Constants.DAILY_EMBED_DIM));
        if (savedWeight.isPresent()) {
            try {
                byte[] weightBytes = savedWeight.get().getWeightBytes();
                MemoryModelSerializer.deserializeFromBytes(model, weightBytes, Constants.DAILY_MODEL_NAME);
                log.info("Successfully loaded Global Model weights from DB (Updated: {}, Size: {} KB)",
                        savedWeight.get().getUpdatedAt(), (weightBytes.length / 1024));
            } catch (Exception e) {
                log.error("Failed to load global model weights from DB: {}", e.getMessage(), e);
            }
        } else {
            log.info("No saved global model found in database. Fresh architecture initialized in RAM.");
        }
    }

    @Override
    @SneakyThrows
    public void trainModel( int sequenceLength, int epochs, float learningRate) {
        List<String> symbols = mongoTemplate.findDistinct("symbol", CandleStick.class, String.class);
        if (symbols.isEmpty()) {
            return;
        }

        log.info("Found {} distinct stock symbols in MongoDB. Aggregating dataset...", symbols.size());

        List<float[][]> allInputs = new ArrayList<>();
        List<float[][]> allTargets = new ArrayList<>();

        for (String symbol : symbols) {
            try (var stream = candleStickRepo.findBySymbolOrderByTimestampAsc(symbol)) {
                DatasetBuilder.TrainingBatch batch = DatasetBuilder.buildSlidingWindows(stream, sequenceLength);
                for (int i = 0; i < batch.inputSequences().length; i++) {
                    allInputs.add(batch.inputSequences()[i]);
                    allTargets.add(batch.targetCandles()[i]);
                }
            } catch (Exception e) {
                log.warn("Skipping symbol {} during dataset assembly: {}", symbol, e.getMessage());
            }
        }

        if (allInputs.isEmpty()) {
            throw new IllegalArgumentException("No dataset samples collected across symbols.");
        }

        int totalSamples = allInputs.size();
        log.info("Dataset aggregated successfully. Total Global Training Samples: {}", totalSamples);

        float[][][] inputData = allInputs.toArray(new float[0][][]);
        float[][][] targetData = allTargets.toArray(new float[0][][]);

        // 3. Train Global Model
        Model globalModel = Model.newInstance(Constants.DAILY_MODEL_NAME);
        globalModel.setBlock(DataTransformerBlock.buildArchitecture(Constants.DAILY_FEATURE_DIM, Constants.DAILY_EMBED_DIM));

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build());

        float finalLoss = 0.0f;

        try (NDManager manager = NDManager.newBaseManager();
             Trainer trainer = globalModel.newTrainer(config)) {

            trainer.initialize(new Shape(1, sequenceLength, Constants.DAILY_FEATURE_DIM));

            NDArray inputTensor = DataTransformerBlock.create3DTensor(manager, inputData);
            NDArray targetTensor = DataTransformerBlock.create3DTensor(manager, targetData);

            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(inputTensor)
                    .optLabels(targetTensor)
                    .setSampling(32, true)
                    .build();

            log.info("Starting Global Model Training | Stocks: {} | Samples: {} | Epochs: {}...",
                    symbols.size(), totalSamples, epochs);

            for (int epoch = 1; epoch <= epochs; epoch++) {
                float accumulatedLoss = 0.0f;
                int batchCount = 0;

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

                finalLoss = accumulatedLoss / Math.max(1, batchCount);
                if (epoch % 5 == 0 || epoch == epochs) {
                    log.info("Global Model | Epoch {}/{} - Avg Batch Loss: {}", epoch, epochs, finalLoss);
                }
            }
        }

        // 4. Save trained weights to MongoDB
        byte[] weightBytes = MemoryModelSerializer.serializeToBytes(globalModel);

        ModelWeightEntity weightEntity = modelWeightRepository.findById(Constants.DAILY_MODEL_NAME)
                .orElseGet(() -> ModelWeightEntity.builder().name(Constants.DAILY_MODEL_NAME).build());

        weightEntity.setWeightBytes(weightBytes);
        weightEntity.setSequenceLength(sequenceLength);
        weightEntity.setFeatureDim(Constants.DAILY_FEATURE_DIM);
        weightEntity.setFinalLoss(finalLoss);
        weightEntity.setUpdatedAt(Instant.now());

        modelWeightRepository.save(weightEntity);
        Constants.DAILY_MODEL = globalModel;

        log.info("Global Model successfully trained on {} stocks and persisted to MongoDB ({} KB).",
                symbols.size(), (weightBytes.length / 1024));
    }

    @Override
    public PredictionResponse predictNextDay(Model trainedModel, String symbol, int sequenceLength) {
        String stockSymbol = symbol.toUpperCase();

        // 1. Extract the LAST 60 candles to serve as the prompt context
        List<CandleStick> last60 = candleStickRepo.findTop60BySymbolOrderByTimestampDesc(symbol).reversed();
        var latestCandle = last60.getLast();
        double basePrice = last60.getFirst().getOpen(); // Anchor base price for local normalization

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
            throw new IllegalStateException("Failed to run DJL inference for symbol: " + symbol, e);
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
