package com.app.oneklikai.util;

import com.app.oneklikai.model.entity.CandleStick;

public class FeatureScaler {

    public static float[] normalizeCandle(CandleStick candle, double basePrice) {
        // Log-scaling balances volume variance with percentage price changes:
        float logVolume = (float) Math.log1p(candle.getVolume()) / 20.0f;
        return new float[]{
                (float) ((candle.getOpen() - basePrice) / basePrice),
                (float) ((candle.getHigh() - basePrice) / basePrice),
                (float) ((candle.getLow() - basePrice) / basePrice),
                (float) ((candle.getClose() - basePrice) / basePrice),
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

    public static float[] normalizeRawCandle(float[] candle, double basePrice) {
        float logVolume = (float) Math.log1p(candle[4]) / 20.0f;
        return new float[]{
                (float) ((candle[0] - basePrice) / basePrice),
                (float) ((candle[1] - basePrice) / basePrice),
                (float) ((candle[2] - basePrice) / basePrice),
                (float) ((candle[3] - basePrice) / basePrice),
                logVolume
        };
    }

}