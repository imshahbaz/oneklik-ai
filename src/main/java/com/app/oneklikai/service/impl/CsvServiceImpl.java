package com.app.oneklikai.service.impl;

import com.app.oneklikai.model.csv.YahooCandleStick;
import com.app.oneklikai.service.CsvService;
import com.app.oneklikai.util.CsvLoader;
import com.app.oneklikai.util.DataCleaner;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStreamReader;
import java.io.Reader;
import java.util.List;

@Service
public class CsvServiceImpl implements CsvService {

    @Override
    public void saveYahooCandleSticks(String symbol, MultipartFile file) {
        if (file.isEmpty()) {
            return;
        }

        try (Reader reader = new InputStreamReader(file.getInputStream())) {
            List<YahooCandleStick> loadedCandles = CsvLoader.loadCsv(reader);
            candleSticks.put(symbol, DataCleaner.prepareDataset(loadedCandles));

        } catch (Exception e) {
        }
    }

}
