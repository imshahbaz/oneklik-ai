package com.app.oneklikai.util;

import com.app.oneklikai.model.csv.YahooCandleStick;

import java.util.Comparator;
import java.util.List;

public class DataCleaner {

    public static List<YahooCandleStick> prepareDataset(List<YahooCandleStick> rawCandles) {
        return rawCandles.stream()
                // 1. Remove invalid rows with bad prices
                .filter(c -> c.open() > 0 && c.high() > 0 && c.low() > 0 && c.close() > 0 && c.volume() > 0)
                // 2. Sort chronologically from oldest to newest
                .sorted(Comparator.comparing(YahooCandleStick::timestamp))
                .toList();
    }
}