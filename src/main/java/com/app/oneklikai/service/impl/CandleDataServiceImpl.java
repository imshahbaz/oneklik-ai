package com.app.oneklikai.service.impl;

import ai.djl.Model;
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
import ai.djl.translate.TranslateException;
import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.model.entity.CandleStick;
import com.app.oneklikai.repo.CandleStickRepo;
import com.app.oneklikai.service.CandleDataService;
import com.app.oneklikai.util.DataTransformerBlock;
import com.app.oneklikai.util.DatasetBuilder;
import com.app.oneklikai.util.FeatureScaler;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CandleDataServiceImpl implements CandleDataService {

    private final CandleStickRepo candleStickRepo;

    @SneakyThrows
    @Override
    public void trainModel(String symbol, int sequenceLength, int epochs, float learningRate) {
        String stockSymbol = symbol.toUpperCase();

        try (var stream = candleStickRepo.findBySymbolOrderByTimestampAsc(symbol)) {
            DatasetBuilder.TrainingBatch batch = DatasetBuilder.buildSlidingWindows(stream, sequenceLength);
            int numSamples = batch.inputSequences().length;
            int featureDim = 5;
            int embedDim = 64;

            // Create DJL Model Container
            try (NDManager manager = NDManager.newBaseManager()) {

                // Set Network Block
                Constants.DAILY_MODEL.setBlock(DataTransformerBlock.buildArchitecture(featureDim, embedDim));

                // Configure Optimizer and Loss Function
                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                        .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build());

                try (Trainer trainer = Constants.DAILY_MODEL.newTrainer(config)) {

                    // Initialize weights based on input tensor shape [BatchSize, SeqLen, Features]
                    trainer.initialize(new Shape(1, sequenceLength, featureDim));

                    // Convert Java float[][][] arrays to Off-Heap Native C++ Tensors
                    NDArray inputTensor = DataTransformerBlock.create3DTensor(manager, batch.inputSequences());
                    NDArray targetTensor = DataTransformerBlock.create3DTensor(manager, batch.targetCandles());

                    System.out.printf("Starting Training on %s | Samples: %d | Epochs: %d...%n", stockSymbol, numSamples, epochs);

                    ArrayDataset dataset = new ArrayDataset.Builder()
                            .setData(inputTensor)
                            .optLabels(targetTensor)
                            .setSampling(32, true) // Batch size = 32, Shuffle = true
                            .build();

                    for (int epoch = 1; epoch <= epochs; epoch++) {
                        float accumulatedLoss = 0.0f;
                        int batchCount = 0;

                        try {
                            for (Batch miniBatch : trainer.iterateDataset(dataset)) {
                                try (GradientCollector gc = trainer.newGradientCollector()) {

                                    // Forward Pass on Mini-Batch
                                    NDArray predictions = trainer.forward(miniBatch.getData()).singletonOrThrow();

                                    // Calculate MSE Loss on Mini-Batch Labels
                                    NDArray loss = trainer.getLoss().evaluate(miniBatch.getLabels(), new NDList(predictions));

                                    // Backward Pass & Step
                                    gc.backward(loss);
                                    trainer.step();

                                    accumulatedLoss += loss.getFloat();
                                    batchCount++;
                                }
                                miniBatch.close(); // Prevent C++ native memory leaks
                            }
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        } catch (TranslateException e) {
                            throw new RuntimeException(e);
                        }

                        if (epoch % 5 == 0 || epoch == epochs) {
                            float avgLoss = accumulatedLoss / Math.max(1, batchCount);
                            System.out.printf("Epoch %d/%d - Avg Batch Loss: %.6f%n", epoch, epochs, avgLoss);
                        }
                    }
                }

            }
        }

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
        try (NDManager manager = NDManager.newBaseManager();
             Trainer trainer = trainedModel.newTrainer(new DefaultTrainingConfig(Loss.l2Loss()))) {

            NDArray inputTensor = DataTransformerBlock.create3DTensor(manager, inputWindow);
            NDArray outputTensor = trainer.forward(new NDList(inputTensor)).singletonOrThrow();

            // Extract the [1, 5] normalized outputs (Open, High, Low, Close, Volume)
            predictedNormalized = outputTensor.get(0).get(0).toFloatArray();
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
