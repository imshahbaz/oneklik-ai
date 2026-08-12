package com.app.oneklikai.model.entity;

import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.csv.YahooCandleStick;
import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.MongoId;

import java.time.Instant;
import java.time.ZoneId;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Document(collection = "candle_sticks")
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CandleStick {

    @MongoId
    String id;
    String symbol;
    Instant timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    TimeFrame timeFrame;

    public YahooCandleStick toYahooCandleStick() {
        return YahooCandleStick.builder()
                .open(open)
                .high(high)
                .low(low)
                .close(close)
                .volume(volume)
                .timestamp(timestamp.atZone(ZoneId.of("Asia/Kolkata")).toOffsetDateTime())
                .timeFrame(timeFrame)
                .build();
    }
}
