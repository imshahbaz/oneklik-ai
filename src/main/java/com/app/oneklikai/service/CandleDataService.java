package com.app.oneklikai.service;

import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.model.dto.response.TrainingReport;

public interface CandleDataService {

    TrainingReport trainModel(String symbol, int sequenceLength, int epochs, float learningRate);

    PredictionResponse predictNextDay(String symbol, int sequenceLength);
}
