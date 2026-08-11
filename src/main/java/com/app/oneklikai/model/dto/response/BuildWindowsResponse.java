package com.app.oneklikai.model.dto.response;

public record BuildWindowsResponse(
    String symbol,
    int totalCandlesInMemory,
    int sequenceLength,
    int totalTrainingSamplesGenerated,
    String inputTensorShape,
    String targetTensorShape,
    float[][] sampleInputWindow,   // First 60-day normalized window sample
    float[][] sampleTargetWindow  // First target candle sample
) {}