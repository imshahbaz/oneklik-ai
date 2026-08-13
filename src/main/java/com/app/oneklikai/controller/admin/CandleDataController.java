package com.app.oneklikai.controller.admin;

import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.service.CandleDataService;
import jakarta.validation.constraints.NotBlank;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/admin/candles")
public class CandleDataController {

    private final CandleDataService candleDataService;

    @PostMapping("/train")
    public void train(@RequestParam("symbol") @NotBlank String symbol) {
        candleDataService.trainModel(symbol, Constants.DAILY_SEQUENCE_LENGTH, Constants.DAILY_EPOCHS, Constants.DAILY_LEARNING_RATE);
    }

    @GetMapping("/predict")
    public PredictionResponse predict(@RequestParam("symbol") String symbol) {
        return candleDataService.predictNextDay(symbol, Constants.DAILY_SEQUENCE_LENGTH);
    }
}