package com.app.oneklikai.service;

import com.app.oneklikai.model.dto.response.PredictionResponse;

public interface CandleDataService {

    void trainModel(String symbol,int sequenceLength, int epochs, float learningRate);

    PredictionResponse predictNextDay(String symbol, int sequenceLength);
}
