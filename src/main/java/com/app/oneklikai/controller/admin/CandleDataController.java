package com.app.oneklikai.controller.admin;

import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.model.dto.response.ApiResponse;
import com.app.oneklikai.model.dto.response.PredictionResponse;
import com.app.oneklikai.model.dto.response.TrainingReport;
import com.app.oneklikai.service.CandleDataService;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/admin/candles")
public class CandleDataController {

    private final CandleDataService candleDataService;

    /**
     * Trains the daily model and returns its out-of-sample scores. The defaults are tuned for a full
     * 20-year daily history; the overrides exist for experimentation.
     */
    @PostMapping("/train")
    public ResponseEntity<ApiResponse<TrainingReport>> train(
            @RequestParam("symbol") @NotBlank String symbol,
            @RequestParam(value = "sequenceLength", required = false) @Positive Integer sequenceLength,
            @RequestParam(value = "epochs", required = false) @Positive Integer epochs,
            @RequestParam(value = "learningRate", required = false) @Positive Float learningRate) {

        TrainingReport report = candleDataService.trainModel(
                symbol,
                sequenceLength != null ? sequenceLength : Constants.DAILY_SEQUENCE_LENGTH,
                epochs != null ? epochs : Constants.DAILY_EPOCHS,
                learningRate != null ? learningRate : Constants.DAILY_LEARNING_RATE);

        return ResponseEntity.ok(ApiResponse.ok(report, "Model trained successfully!"));
    }

    @GetMapping("/predict")
    public ResponseEntity<ApiResponse<PredictionResponse>> predict(@RequestParam("symbol") String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(candleDataService.predictNextDay(symbol, Constants.DAILY_SEQUENCE_LENGTH), "Prediction successful!"));
    }
}
