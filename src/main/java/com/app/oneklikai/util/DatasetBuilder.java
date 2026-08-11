package com.app.oneklikai.util;

import com.app.oneklikai.model.csv.YahooCandleStick;

import java.util.List;

public class DatasetBuilder {

    public record TrainingBatch(
        float[][][] inputSequences, // Dimensions: [NumSamples, 60, 5]
        float[][][] targetCandles   // Dimensions: [NumSamples, 1, 5]
    ) {}

    public static TrainingBatch buildSlidingWindows(List<YahooCandleStick> candles, int sequenceLength) {
        int totalSamples = candles.size() - sequenceLength;
        if (totalSamples <= 0) {
            throw new IllegalArgumentException("Not enough candles to create sliding windows.");
        }

        int featureDim = 5; // Open, High, Low, Close, Volume
        float[][][] inputs = new float[totalSamples][sequenceLength][featureDim];
        float[][][] targets = new float[totalSamples][1][featureDim];

        for (int i = 0; i < totalSamples; i++) {
            // Anchor base price for this specific window
            double basePrice = candles.get(i).open();

            // 1. Build 60-day historical window
            for (int j = 0; j < sequenceLength; j++) {
                inputs[i][j] = FeatureScaler.normalizeCandle(candles.get(i + j), basePrice);
            }

            // 2. Set the 61st candle as ground truth target
            targets[i][0] = FeatureScaler.normalizeCandle(candles.get(i + sequenceLength), basePrice);
        }

        return new TrainingBatch(inputs, targets);
    }
}