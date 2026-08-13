package com.app.oneklikai.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.loss.Loss;

/**
 * Huber-style regression loss with per-output and per-sample weighting.
 *
 * <p>Three deliberate differences from the plain {@code L2Loss} this replaces:
 *
 * <ul>
 *   <li><b>Huber instead of squared error.</b> Two decades of daily candles contain circuit-breaker
 *       days and gap opens whose squared error dwarfs everything else; L2 spends the whole model
 *       capacity fitting those outliers. The smooth (pseudo-)Huber form is quadratic near zero and
 *       linear in the tails, so ordinary days keep driving the gradient.
 *   <li><b>Per-output weights.</b> The close is what the prediction is judged on, volume is nearly
 *       noise; weighting them equally wastes capacity. See {@link FeatureScaler#LOSS_WEIGHTS}.
 *   <li><b>Per-sample weights</b>, read from the last label column, so recent years can count for
 *       more than 2006 without discarding old data.
 * </ul>
 */
public class WeightedHuberLoss extends Loss {

    private final float delta;
    private final float[] outputWeights;

    public WeightedHuberLoss(String name, float delta, float[] outputWeights) {
        super(name);
        this.delta = delta;
        this.outputWeights = outputWeights.clone();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray prediction = predictions.singletonOrThrow();
        NDArray label = labels.singletonOrThrow();

        NDArray target = label.get(new NDIndex(":, :, 0:{}", FeatureScaler.TARGET_DIM));
        NDArray sampleWeight = label.get(
                new NDIndex(":, :, {}:{}", FeatureScaler.WEIGHT_INDEX, FeatureScaler.LABEL_DIM));

        NDArray error = target.sub(prediction.reshape(target.getShape())).div(delta);
        // Pseudo-Huber: delta^2 * (sqrt(1 + (e/delta)^2) - 1) -- smooth everywhere, no branching.
        NDArray elementWise = error.square().add(1.0f).sqrt().sub(1.0f).mul(delta * delta);

        NDArray weights = prediction.getManager().create(outputWeights);
        return elementWise.mul(weights).mul(sampleWeight).mean();
    }
}
