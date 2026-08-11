package com.app.oneklikai.controller.admin;

import com.app.oneklikai.service.CsvService;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/admin/csv")
public class CsvController {

    private final CsvService csvService;

    @PostMapping(value = "/yahoo/{symbol}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void saveYahooCandleSticks(@PathVariable @NotNull String symbol, @RequestParam("file") @NotNull MultipartFile file) {
        csvService.saveYahooCandleSticks(symbol, file);
        System.out.println(CsvService.candleSticks.get(symbol).size());
    }
}
