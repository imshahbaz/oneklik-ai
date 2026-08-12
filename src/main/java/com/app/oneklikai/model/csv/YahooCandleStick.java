package com.app.oneklikai.model.csv;

import com.app.oneklikai.model.entity.CandleStick;
import lombok.Builder;

import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

@Builder
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

    public CandleStick toEntity(String symbol){
        return CandleStick.builder()
                .symbol(symbol)
                .open(open)
                .high(high)
                .low(low)
                .close(close)
                .volume(volume)
                .timestamp(this.timestamp.atZoneSameInstant(ZoneId.of("Asia/Kolkata")).toInstant())
                .build();
    }
}