package com.app.oneklikai.service;

import com.app.oneklikai.model.TimeFrame;
import org.springframework.web.multipart.MultipartFile;

public interface CsvService {

    void saveYahooCandleSticks(String symbol, MultipartFile file, TimeFrame timeFrame);
}
