package com.app.oneklikai.util;

import com.app.oneklikai.model.entity.CandleStick;

import java.time.ZoneId;
import java.util.List;

/**
 * Turns raw OHLCV candles into a causal, scale-free feature matrix.
 *
 * <p>Every feature is a ratio (log price ratio, oscillator, range fraction), so a candle from 2005
 * at ₹50 and a candle from 2025 at ₹5000 produce comparable numbers. That is what makes 20 years of
 * history usable as one training distribution instead of three unrelated price regimes. Because the
 * features carry no absolute price level, no dataset-wide scaler statistics have to be persisted
 * alongside the weights: inference recomputes exactly the same numbers from the last candles.
 *
 * <p>All indicators are computed strictly from past-and-current candles (never forward looking), and
 * each one falls back to an expanding window while its period is still filling up, so only
 * {@link #WARMUP} candles are discarded instead of 200.
 */
public final class FeatureEngineer {

    /** Bumped whenever the feature layout changes; persisted weights of an older version are ignored. */
    public static final int FEATURE_VERSION = 2;

    /** Number of input features per timestep. */
    public static final int FEATURE_DIM = 23;

    /** Candles dropped at the head of the series so indicators are meaningful. */
    public static final int WARMUP = 60;

    private static final ZoneId MARKET_ZONE = ZoneId.of("Asia/Kolkata");

    // Fixed multipliers that bring each ratio into roughly unit variance for a daily equity series.
    private static final double DAILY_RETURN_SCALE = 50.0;
    private static final double SHORT_TREND_SCALE = 20.0;
    private static final double MID_TREND_SCALE = 10.0;
    private static final double LONG_TREND_SCALE = 8.0;
    private static final double VERY_LONG_TREND_SCALE = 5.0;
    private static final double MACD_SCALE = 100.0;
    private static final double FEATURE_CLIP = 10.0;

    private FeatureEngineer() {
    }

    /** Column-major view of a chronologically sorted candle series. */
    public record Series(
            double[] open,
            double[] high,
            double[] low,
            double[] close,
            double[] volume,
            int[] dayOfWeek
    ) {
        public int length() {
            return close.length;
        }
    }

    /**
     * @param features   {@code [length][FEATURE_DIM]} feature rows, index aligned with the series
     * @param volumeEma  20-period EMA of volume, used as the scale-free anchor for volume targets
     */
    public record FeatureMatrix(float[][] features, double[] volumeEma) {
    }

    public static Series toSeries(List<CandleStick> candles) {
        int n = candles.size();
        double[] open = new double[n];
        double[] high = new double[n];
        double[] low = new double[n];
        double[] close = new double[n];
        double[] volume = new double[n];
        int[] dayOfWeek = new int[n];

        for (int i = 0; i < n; i++) {
            CandleStick c = candles.get(i);
            open[i] = c.getOpen();
            high[i] = Math.max(c.getHigh(), Math.max(c.getOpen(), c.getClose()));
            low[i] = Math.min(c.getLow(), Math.min(c.getOpen(), c.getClose()));
            close[i] = c.getClose();
            volume[i] = Math.max(c.getVolume(), 0.0);
            dayOfWeek[i] = c.getTimestamp().atZone(MARKET_ZONE).getDayOfWeek().getValue();
        }

        return new Series(open, high, low, close, volume, dayOfWeek);
    }

    public static FeatureMatrix compute(Series s) {
        int n = s.length();
        double[] open = s.open();
        double[] high = s.high();
        double[] low = s.low();
        double[] close = s.close();
        double[] volume = s.volume();

        double[] logReturn = new double[n];
        for (int i = 1; i < n; i++) {
            logReturn[i] = safeLog(close[i], close[i - 1]);
        }

        double[] sma5 = sma(close, 5);
        double[] sma10 = sma(close, 10);
        double[] sma20 = sma(close, 20);
        double[] sma50 = sma(close, 50);
        double[] sma200 = sma(close, 200);
        double[] volatility20 = rollingStd(logReturn, 20);
        double[] rsi14 = wilderRsi(close, 14);
        double[] atr14 = wilderAtr(high, low, close, 14);
        double[] volumeEma = ema(volume, 20);
        double[] macdHistogram = macdHistogram(close);

        float[][] features = new float[n][FEATURE_DIM];
        for (int i = 1; i < n; i++) {
            double prevClose = close[i - 1];
            double range = high[i] - low[i];
            double[] row = new double[FEATURE_DIM];

            // Candle shape relative to yesterday's close: the core short-horizon signal.
            row[0] = DAILY_RETURN_SCALE * safeLog(open[i], prevClose);
            row[1] = DAILY_RETURN_SCALE * safeLog(high[i], prevClose);
            row[2] = DAILY_RETURN_SCALE * safeLog(low[i], prevClose);
            row[3] = DAILY_RETURN_SCALE * logReturn[i];

            // Intraday range and where the close sat inside it (buying vs selling pressure).
            row[4] = DAILY_RETURN_SCALE * safeLog(high[i], low[i]) - 1.0;
            row[5] = range > 0 ? (close[i] - low[i]) / range - 0.5 : 0.0;
            row[6] = range > 0 ? (close[i] - open[i]) / range : 0.0;

            // Volume relative to its own recent level, log-compressed.
            row[7] = Math.log((volume[i] + 1.0) / (volumeEma[i] + 1.0));

            // Distance from moving averages: multi-horizon trend context.
            row[8] = SHORT_TREND_SCALE * safeLog(close[i], sma5[i]);
            row[9] = SHORT_TREND_SCALE * safeLog(close[i], sma10[i]);
            row[10] = MID_TREND_SCALE * safeLog(close[i], sma20[i]);
            row[11] = LONG_TREND_SCALE * safeLog(close[i], sma50[i]);
            row[12] = VERY_LONG_TREND_SCALE * safeLog(close[i], sma200[i]);

            // Volatility regime: lets the model widen or tighten its predicted range.
            row[13] = DAILY_RETURN_SCALE * volatility20[i] - 1.0;
            row[14] = close[i] > 0 ? DAILY_RETURN_SCALE * (atr14[i] / close[i]) - 1.0 : 0.0;

            // Oscillators.
            row[15] = rsi14[i] / 50.0 - 1.0;
            row[16] = close[i] > 0 ? MACD_SCALE * (macdHistogram[i] / close[i]) : 0.0;
            row[17] = SHORT_TREND_SCALE * safeLog(sma10[i], sma50[i]);

            // Momentum over 1 week / 1 month / 1 quarter.
            row[18] = SHORT_TREND_SCALE * safeLog(close[i], close[Math.max(0, i - 5)]);
            row[19] = MID_TREND_SCALE * safeLog(close[i], close[Math.max(0, i - 20)]);
            row[20] = VERY_LONG_TREND_SCALE * safeLog(close[i], close[Math.max(0, i - 60)]);

            // Weekday seasonality (Monday gaps, Friday squaring off).
            double weekdayAngle = 2.0 * Math.PI * s.dayOfWeek()[i] / 7.0;
            row[21] = Math.sin(weekdayAngle);
            row[22] = Math.cos(weekdayAngle);

            for (int f = 0; f < FEATURE_DIM; f++) {
                features[i][f] = sanitize(row[f]);
            }
        }

        return new FeatureMatrix(features, volumeEma);
    }

    private static double[] sma(double[] x, int period) {
        double[] out = new double[x.length];
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i];
            if (i >= period) {
                sum -= x[i - period];
            }
            out[i] = sum / Math.min(i + 1, period);
        }
        return out;
    }

    private static double[] ema(double[] x, int period) {
        double[] out = new double[x.length];
        if (x.length == 0) {
            return out;
        }
        double alpha = 2.0 / (period + 1.0);
        out[0] = x[0];
        for (int i = 1; i < x.length; i++) {
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
        }
        return out;
    }

    /** Standard deviation over the trailing {@code period}, expanding while the window fills. */
    private static double[] rollingStd(double[] x, int period) {
        double[] out = new double[x.length];
        double sum = 0.0;
        double sumSquares = 0.0;
        for (int i = 1; i < x.length; i++) {
            sum += x[i];
            sumSquares += x[i] * x[i];
            if (i > period) {
                double dropped = x[i - period];
                sum -= dropped;
                sumSquares -= dropped * dropped;
            }
            int count = Math.min(i, period);
            double mean = sum / count;
            out[i] = Math.sqrt(Math.max(0.0, sumSquares / count - mean * mean));
        }
        return out;
    }

    private static double[] wilderRsi(double[] close, int period) {
        double[] out = new double[close.length];
        double avgGain = 0.0;
        double avgLoss = 0.0;
        if (close.length > 0) {
            out[0] = 50.0;
        }
        for (int i = 1; i < close.length; i++) {
            double change = close[i] - close[i - 1];
            avgGain += (Math.max(change, 0.0) - avgGain) / period;
            avgLoss += (Math.max(-change, 0.0) - avgLoss) / period;
            double total = avgGain + avgLoss;
            out[i] = total > 0 ? 100.0 * avgGain / total : 50.0;
        }
        return out;
    }

    private static double[] wilderAtr(double[] high, double[] low, double[] close, int period) {
        double[] out = new double[close.length];
        if (close.length == 0) {
            return out;
        }
        out[0] = high[0] - low[0];
        for (int i = 1; i < close.length; i++) {
            double trueRange = Math.max(high[i] - low[i],
                    Math.max(Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1])));
            out[i] = out[i - 1] + (trueRange - out[i - 1]) / period;
        }
        return out;
    }

    private static double[] macdHistogram(double[] close) {
        double[] fast = ema(close, 12);
        double[] slow = ema(close, 26);
        double[] macd = new double[close.length];
        for (int i = 0; i < close.length; i++) {
            macd[i] = fast[i] - slow[i];
        }
        double[] signal = ema(macd, 9);
        double[] histogram = new double[close.length];
        for (int i = 0; i < close.length; i++) {
            histogram[i] = macd[i] - signal[i];
        }
        return histogram;
    }

    /** Natural log of a ratio, defensive against zero/negative prices in dirty historical data. */
    private static double safeLog(double numerator, double denominator) {
        if (numerator <= 0 || denominator <= 0) {
            return 0.0;
        }
        return Math.log(numerator / denominator);
    }

    private static float sanitize(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0f;
        }
        return (float) Math.clamp(value, -FEATURE_CLIP, FEATURE_CLIP);
    }
}
