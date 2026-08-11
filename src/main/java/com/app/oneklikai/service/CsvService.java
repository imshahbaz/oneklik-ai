package com.app.oneklikai.service;

import com.app.oneklikai.model.csv.YahooCandleStick;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public interface CsvService {

    Map<String, List<YahooCandleStick>> candleSticks = new ConcurrentHashMap<>();

    void saveYahooCandleSticks(String symbol, MultipartFile file);
}
