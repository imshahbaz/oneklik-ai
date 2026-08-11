package com.app.oneklikai.util;

import com.app.oneklikai.model.csv.YahooCandleStick;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

public class CsvLoader {

    public static List<YahooCandleStick> loadCsv(Reader reader) throws Exception {
        List<YahooCandleStick> candles = new ArrayList<>();

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true).get();

        try (CSVParser csvParser = csvFormat.parse(reader)) {
            for (CSVRecord record : csvParser) {
                OffsetDateTime timestamp = OffsetDateTime.parse(record.get("Date"), YahooCandleStick.FORMATTER);
                double open = Double.parseDouble(record.get("Open"));
                double high = Double.parseDouble(record.get("High"));
                double low = Double.parseDouble(record.get("Low"));
                double close = Double.parseDouble(record.get("Close"));
                double volume = Double.parseDouble(record.get("Volume"));

                candles.add(new YahooCandleStick(timestamp, open, high, low, close, volume));
            }
        }

        return candles;
    }

}