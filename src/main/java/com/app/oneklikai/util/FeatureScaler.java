package com.app.oneklikai.util;

import com.app.oneklikai.model.csv.YahooCandleStick;

public class FeatureScaler {

    public static float[] normalizeCandle(YahooCandleStick candle, double basePrice) {
        // Log-scaling balances volume variance with percentage price changes:
        float logVolume = (float) Math.log1p(candle.volume()) / 20.0f;
        return new float[]{
                (float) ((candle.open() - basePrice) / basePrice),
                (float) ((candle.high() - basePrice) / basePrice),
                (float) ((candle.low() - basePrice) / basePrice),
                (float) ((candle.close() - basePrice) / basePrice),
                logVolume
        };
    }

    public static double denormalizePrice(double normalizedPrice, double basePrice) {
        return (normalizedPrice * basePrice) + basePrice;
    }

    public static double denormalizeVolume(double normalizedLogVolume) {
        double scaledLog = normalizedLogVolume * 20.0;
        return Math.expm1(scaledLog);
    }
}