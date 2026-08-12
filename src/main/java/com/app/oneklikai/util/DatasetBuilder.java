package com.app.oneklikai.util;

import com.app.oneklikai.model.entity.CandleStick;

import java.util.List;
import java.util.stream.Stream;

public class DatasetBuilder {

    public record TrainingBatch(
            float[][][] inputSequences, // Dimensions: [NumSamples, 60, 5]
            float[][][] targetCandles   // Dimensions: [NumSamples, 1, 5]
    ) {
    }

    public static TrainingBatch buildSlidingWindows(Stream<CandleStick> candleStream, int sequenceLength) {
        int featureDim = 5;

        // 1. Collect stream into lightweight 2D float matrix [TotalCandles, 5]
        List<float[]> rawMatrixList = candleStream
                .map(c -> new float[]{
                        (float) c.getOpen(),
                        (float) c.getHigh(),
                        (float) c.getLow(),
                        (float) c.getClose(),
                        (float) c.getVolume(),
                })
                .toList();

        int totalCandles = rawMatrixList.size();
        int totalSamples = totalCandles - sequenceLength;

        if (totalSamples <= 0) {
            throw new IllegalArgumentException(
                    "Not enough candles to create sliding windows. Found: " + totalCandles + ", Required > " + sequenceLength
            );
        }

        float[][][] inputs = new float[totalSamples][sequenceLength][featureDim];
        float[][][] targets = new float[totalSamples][1][featureDim];

        // 2. Build sliding windows using simple array offsets
        for (int i = 0; i < totalSamples; i++) {
            // Anchor base price for this window (first open price in the window)
            double basePrice = rawMatrixList.get(i)[0];

            // Build sequence window [sequenceLength, 5]
            for (int j = 0; j < sequenceLength; j++) {
                float[] candle = rawMatrixList.get(i + j);
                inputs[i][j] = FeatureScaler.normalizeRawCandle(candle, basePrice);
            }

            // Set the next candle as target [1, 5]
            float[] targetCandle = rawMatrixList.get(i + sequenceLength);
            targets[i][0] = FeatureScaler.normalizeRawCandle(targetCandle, basePrice);
        }

        return new TrainingBatch(inputs, targets);
    }
}