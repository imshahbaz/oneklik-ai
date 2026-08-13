package com.app.oneklikai.util;

/**
 * Encodes/decodes the prediction target.
 *
 * <p>The target is the <b>log change of the next candle relative to the last close of the input
 * window</b> (the "anchor"), not a normalised absolute price. Two consequences matter for accuracy:
 *
 * <ul>
 *   <li>The target is stationary — a 1% move in 2006 and a 1% move in 2026 are the same label, so
 *       every one of the ~5000 windows in 20 years teaches the same function.
 *   <li>The anchor is the <i>last</i> observed close rather than the first open of the window, so the
 *       model predicts a one-step-ahead change instead of a 60-day-cumulative drift. That removes the
 *       large, mostly-unpredictable trend component from the label and leaves the part that is
 *       actually learnable.
 * </ul>
 *
 * <p>Log returns are multiplied by {@link #PRICE_SCALE} so the labels sit near unit magnitude, which
 * keeps gradients healthy without changing what is being learned.
 */
public final class FeatureScaler {

    /** open, high, low, close, volume. */
    public static final int TARGET_DIM = 5;

    /** Targets carry one extra trailing column: the per-sample training weight. */
    public static final int LABEL_DIM = TARGET_DIM + 1;

    public static final int CLOSE_INDEX = 3;
    public static final int WEIGHT_INDEX = TARGET_DIM;

    private static final float PRICE_SCALE = 20.0f;
    private static final float VOLUME_SCALE = 1.0f;

    /** A single day cannot plausibly move more than this in log space (~ +/- 60%). */
    private static final double MAX_LOG_MOVE = 0.5;

    /** Relative importance of each output in the loss: the close is what predictions are judged on. */
    public static final float[] LOSS_WEIGHTS = {0.8f, 0.8f, 0.8f, 1.6f, 0.2f};

    private FeatureScaler() {
    }

    /**
     * Encodes the candle following the input window into a label row.
     *
     * @param anchorClose  close of the last candle inside the input window
     * @param anchorVolume volume EMA at the last candle inside the input window
     * @param sampleWeight training weight for this sample (1.0 for evaluation splits)
     */
    public static float[] encodeTarget(double anchorClose,
                                      double anchorVolume,
                                      double open,
                                      double high,
                                      double low,
                                      double close,
                                      double volume,
                                      double sampleWeight) {
        return new float[]{
                logRatio(open, anchorClose) * PRICE_SCALE,
                logRatio(high, anchorClose) * PRICE_SCALE,
                logRatio(low, anchorClose) * PRICE_SCALE,
                logRatio(close, anchorClose) * PRICE_SCALE,
                (float) Math.log((volume + 1.0) / (anchorVolume + 1.0)) * VOLUME_SCALE,
                (float) sampleWeight
        };
    }

    public static double decodePrice(float normalized, double anchorClose) {
        double logMove = Math.clamp(normalized / PRICE_SCALE, -MAX_LOG_MOVE, MAX_LOG_MOVE);
        return anchorClose * Math.exp(logMove);
    }

    public static double decodeVolume(float normalized, double anchorVolume) {
        double logRatio = Math.clamp(normalized / VOLUME_SCALE, -MAX_LOG_MOVE * 10, MAX_LOG_MOVE * 10);
        return Math.max(0.0, (anchorVolume + 1.0) * Math.exp(logRatio) - 1.0);
    }

    /** Converts a scaled log return back to a plain percentage, for human-readable metrics. */
    public static double toPercent(double scaledLogReturn) {
        return (Math.exp(scaledLogReturn / PRICE_SCALE) - 1.0) * 100.0;
    }

    private static float logRatio(double value, double anchor) {
        if (value <= 0 || anchor <= 0) {
            return 0.0f;
        }
        return (float) Math.log(value / anchor);
    }
}
