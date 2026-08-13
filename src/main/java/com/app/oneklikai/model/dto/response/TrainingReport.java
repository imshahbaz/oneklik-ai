package com.app.oneklikai.model.dto.response;

import java.time.OffsetDateTime;
import java.util.List;

/**
 * What the training run actually achieved, so a model is never trusted on faith.
 *
 * <p>{@code test} is measured on the most recent slice of history, which the model never saw during
 * training or early stopping. Compare it against {@code naiveTestBaseline} — the best <b>constant</b>
 * predictor, i.e. the per-head training means with the better of the two constant direction guesses.
 * If directional accuracy is not clearly above that baseline, the model has learned nothing tradable no
 * matter how small the loss looks.
 *
 * <p>Bear the sample size in mind when reading the gap: with ~750 test days the standard error on
 * directional accuracy is around ±1.8 percentage points, so anything under a couple of points is noise.
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
