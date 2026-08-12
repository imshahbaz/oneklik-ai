package com.app.oneklikai.util;

import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.csv.YahooCandleStick;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.time.OffsetDateTime;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class CsvLoader {

    public static Stream<YahooCandleStick> streamCsv(Reader reader, TimeFrame timeFrame) throws Exception {
        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .get();

        CSVParser csvParser = csvFormat.parse(reader);

        // Map CSVParser iterator lazily into a Java Stream
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(csvParser.iterator(), Spliterator.ORDERED),
                false
        ).onClose(() -> {
            try {
                csvParser.close();
            } catch (Exception ignored) {
            }
        }).map(record -> parseRecord(record, timeFrame));
    }

    private static YahooCandleStick parseRecord(CSVRecord record, TimeFrame timeFrame) {
        OffsetDateTime timestamp = OffsetDateTime.parse(record.get("Date"), YahooCandleStick.FORMATTER);
        double open = Double.parseDouble(record.get("Open"));
        double high = Double.parseDouble(record.get("High"));
        double low = Double.parseDouble(record.get("Low"));
        double close = Double.parseDouble(record.get("Close"));
        double volume = Double.parseDouble(record.get("Volume"));

        return new YahooCandleStick(timestamp, open, high, low, close, volume, timeFrame);
    }
}