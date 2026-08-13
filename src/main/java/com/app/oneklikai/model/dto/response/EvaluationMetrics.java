package com.app.oneklikai.model.dto.response;

/**
 * Out-of-sample quality of a trained model on one split.
 *
 * @param loss                        weighted Huber loss in scaled log-return space (early-stopping criterion)
 * @param closeMaePercent             mean absolute error of the next close, in percentage points
 * @param closeRmsePercent            root mean squared error of the next close, in percentage points
 * @param directionalAccuracyPercent  share of days where the predicted up/down move was correct
 * @param samples                     number of evaluated windows
 */
public record EvaluationMetrics(
        double loss,
        double closeMaePercent,
        double closeRmsePercent,
        double directionalAccuracyPercent,
        int samples
) {
}
