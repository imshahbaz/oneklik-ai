package com.app.oneklikai.model.dto.response;

import java.time.OffsetDateTime;

public record PredictionResponse(
    String symbol,
    OffsetDateTime inputLastDate,
    double predictedOpen,
    double predictedHigh,
    double predictedLow,
    double predictedClose,
    double predictedVolume,
    String expectedDirection,
    double expectedReturnPercent
) {}