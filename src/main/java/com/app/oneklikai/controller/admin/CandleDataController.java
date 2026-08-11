package com.app.oneklikai.controller.admin;

import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.service.CandleDataService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/admin/candles")
public class CandleDataController {

    private final CandleDataService candleDataService;

    @PostMapping("/train")
    public void train(
            @RequestParam("symbol") String symbol,
            @RequestParam(name = "sequenceLength", defaultValue = "60") int sequenceLength,
            @RequestParam(name = "epochs", defaultValue = "40") int epochs,
            @RequestParam(name = "learningRate", defaultValue = "0.0005") float learningRate) {
        candleDataService.trainModel(symbol, sequenceLength, epochs, learningRate);
    }

    @GetMapping("/predict")
    public PredictionResponse predict(
            @RequestParam("symbol") String symbol,
            @RequestParam(name = "sequenceLength", defaultValue = "60") int sequenceLength) {
        return candleDataService.predictNextDay(CandleDataService.model, symbol, sequenceLength);
    }
}