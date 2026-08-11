package com.app.oneklikai.model.csv;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;

public record YahooCandleStick(
    OffsetDateTime timestamp,
    double open,
    double high,
    double low,
    double close,
    double volume
) {
    public static final DateTimeFormatter FORMATTER = 
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssXXX");
}