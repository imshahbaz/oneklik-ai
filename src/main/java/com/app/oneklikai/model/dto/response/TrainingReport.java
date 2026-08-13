package com.app.oneklikai.model.dto.response;

import java.time.OffsetDateTime;
import java.util.List;

/**
 * What the training run actually achieved, so a model is never trusted on faith.
 *
 * <p>{@code test} is measured on the most recent slice of history, which the model never saw during
 * training or early stopping. Compare it against {@code naiveTestBaseline} (predict "no change", and
 * always guess the majority direction): if directional accuracy is not clearly above the baseline, the
 * model has learned nothing tradable no matter how small the loss looks.
 */
public record TrainingReport(
        String symbol,
        OffsetDateTime trainedAt,
        int totalCandles,
        int sequenceLength,
        int featureDim,
        int trainSamples,
        int validationSamples,
        int testSamples,
        int restarts,
        int ensembleSize,
        long bestSeed,
        int bestEpoch,
        List<Integer> epochsRunPerRestart,
        EvaluationMetrics validation,
        EvaluationMetrics test,
        EvaluationMetrics naiveTestBaseline,
        int modelSizeKb
) {
}
