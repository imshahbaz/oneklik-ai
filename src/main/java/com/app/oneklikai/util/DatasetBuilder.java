package com.app.oneklikai.util;

import com.app.oneklikai.model.entity.CandleStick;

import java.util.List;

/**
 * Builds sliding-window training samples and splits them <b>chronologically</b>.
 *
 * <p>Why the split shape matters more than any hyperparameter here: with a random split, windows that
 * overlap by 59 of 60 candles land on both sides, the validation loss looks great, and the model is
 * useless on tomorrow. So the series is cut in time order — train on the oldest data, validate on the
 * next slice, test on the most recent slice — and a gap of {@code sequenceLength} samples is
 * <i>purged</i> at each boundary so no training window shares a single candle with an evaluation
 * window. The reported validation/test numbers are then honest estimates of live performance, which is
 * what makes early stopping and model selection meaningful.
 *
 * <p>Training samples also carry an exponential recency weight, so the last few years dominate the fit
 * without throwing away the older regimes that teach the model what a crash or a sideways decade looks
 * like.
 */
public class DatasetBuilder {

    private DatasetBuilder() {
    }

    /**
     * @param inputs {@code [samples][sequenceLength][FeatureEngineer.FEATURE_DIM]}
     * @param labels {@code [samples][1][FeatureScaler.LABEL_DIM]}
     */
    public record WindowSet(float[][][] inputs, float[][][] labels) {
        public int size() {
            return inputs.length;
        }
    }

    public record SplitDataset(
            WindowSet train,
            WindowSet validation,
            WindowSet test,
            int totalCandles,
            int totalSamples
    ) {
    }

    /**
     * @param recencyHalfLifeSamples number of samples over which a sample's training weight halves;
     *                               non-positive disables recency weighting
     */
    public static SplitDataset build(List<CandleStick> candles,
                                     int sequenceLength,
                                     double validationFraction,
                                     double testFraction,
                                     double recencyHalfLifeSamples) {
        FeatureEngineer.Series series = FeatureEngineer.toSeries(candles);
        FeatureEngineer.FeatureMatrix matrix = FeatureEngineer.compute(series);

        int totalCandles = series.length();
        int firstWindowStart = FeatureEngineer.WARMUP;
        int totalSamples = totalCandles - firstWindowStart - sequenceLength;

        int purge = sequenceLength;
        int minimumSamples = 4 * purge + 100;
        if (totalSamples < minimumSamples) {
            throw new IllegalArgumentException(
                    "Not enough candles for a purged chronological split. Candles: " + totalCandles
                            + ", usable samples: " + Math.max(totalSamples, 0)
                            + ", required samples: " + minimumSamples
                            + " (needs roughly " + (firstWindowStart + sequenceLength + minimumSamples)
                            + " candles for sequenceLength=" + sequenceLength + ")");
        }

        int testCount = (int) Math.round(totalSamples * testFraction);
        int validationCount = (int) Math.round(totalSamples * validationFraction);
        int trainCount = totalSamples - testCount - validationCount - 2 * purge;
        if (trainCount <= 0) {
            throw new IllegalArgumentException("Validation/test fractions leave no training samples.");
        }

        int trainFrom = 0;
        int validationFrom = trainCount + purge;
        int testFrom = validationFrom + validationCount + purge;

        WindowSet train = materialize(series, matrix, sequenceLength, firstWindowStart,
                trainFrom, trainCount, trainCount, recencyHalfLifeSamples);
        WindowSet validation = materialize(series, matrix, sequenceLength, firstWindowStart,
                validationFrom, validationCount, trainCount, -1);
        WindowSet test = materialize(series, matrix, sequenceLength, firstWindowStart,
                testFrom, testCount, trainCount, -1);

        return new SplitDataset(train, validation, test, totalCandles, totalSamples);
    }

    /**
     * Materializes samples {@code [from, from + count)} of the global sample space.
     *
     * @param trainEnd              index of the newest training sample, the anchor for recency weights
     * @param recencyHalfLifeSamples negative to weight every sample equally
     */
    private static WindowSet materialize(FeatureEngineer.Series series,
                                         FeatureEngineer.FeatureMatrix matrix,
                                         int sequenceLength,
                                         int firstWindowStart,
                                         int from,
                                         int count,
                                         int trainEnd,
                                         double recencyHalfLifeSamples) {
        float[][][] inputs = new float[count][][];
        float[][][] labels = new float[count][1][];
        float[][] features = matrix.features();
        double[] volumeEma = matrix.volumeEma();

        double weightSum = 0.0;
        for (int s = 0; s < count; s++) {
            int sampleIndex = from + s;
            int windowStart = firstWindowStart + sampleIndex;
            int anchorIndex = windowStart + sequenceLength - 1;
            int targetIndex = anchorIndex + 1;

            float[][] window = new float[sequenceLength][];
            for (int t = 0; t < sequenceLength; t++) {
                window[t] = features[windowStart + t];
            }
            inputs[s] = window;

            double weight = 1.0;
            if (recencyHalfLifeSamples > 0) {
                double ageInSamples = Math.max(0, trainEnd - sampleIndex);
                weight = Math.pow(0.5, ageInSamples / recencyHalfLifeSamples);
            }
            weightSum += weight;

            labels[s][0] = FeatureScaler.encodeTarget(
                    series.close()[anchorIndex],
                    volumeEma[anchorIndex],
                    series.open()[targetIndex],
                    series.high()[targetIndex],
                    series.low()[targetIndex],
                    series.close()[targetIndex],
                    series.volume()[targetIndex],
                    weight);
        }

        // Renormalise to a mean weight of 1 so the loss magnitude stays comparable across splits.
        if (recencyHalfLifeSamples > 0 && count > 0 && weightSum > 0) {
            float normalizer = (float) (count / weightSum);
            for (float[][] label : labels) {
                label[0][FeatureScaler.WEIGHT_INDEX] *= normalizer;
            }
        }

        return new WindowSet(inputs, labels);
    }

    /**
     * Builds the single most recent input window for inference, using exactly the same feature code
     * path as training.
     *
     * @param candles chronologically sorted candles ending with the latest one
     * @return the inference window plus the anchor values needed to decode the prediction
     */
    public static InferenceWindow buildLatestWindow(List<CandleStick> candles, int sequenceLength) {
        FeatureEngineer.Series series = FeatureEngineer.toSeries(candles);
        int totalCandles = series.length();
        int required = FeatureEngineer.WARMUP + sequenceLength;
        if (totalCandles < required) {
            throw new IllegalArgumentException("Need at least " + required
                    + " candles to build an inference window, found " + totalCandles);
        }

        FeatureEngineer.FeatureMatrix matrix = FeatureEngineer.compute(series);
        int windowStart = totalCandles - sequenceLength;
        int anchorIndex = totalCandles - 1;

        float[][][] input = new float[1][sequenceLength][];
        for (int t = 0; t < sequenceLength; t++) {
            input[0][t] = matrix.features()[windowStart + t];
        }

        return new InferenceWindow(
                input,
                series.close()[anchorIndex],
                matrix.volumeEma()[anchorIndex],
                candles.get(anchorIndex));
    }

    public record InferenceWindow(
            float[][][] input,
            double anchorClose,
            double anchorVolumeEma,
            CandleStick anchorCandle
    ) {
    }
}
