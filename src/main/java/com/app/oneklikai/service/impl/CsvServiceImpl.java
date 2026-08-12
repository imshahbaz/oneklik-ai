package com.app.oneklikai.service.impl;

import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.entity.CandleStick;
import com.app.oneklikai.service.CsvService;
import com.app.oneklikai.util.CsvLoader;
import com.app.oneklikai.util.DataCleaner;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.mongodb.core.BulkOperations;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.mongodb.core.query.Update;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class CsvServiceImpl implements CsvService {

    private final MongoTemplate mongoTemplate;

    @Override
    public void saveYahooCandleSticks(String symbol, MultipartFile file, TimeFrame timeFrame) {
        if (file.isEmpty()) {
            return;
        }

        final int BATCH_SIZE = 500;
        List<CandleStick> batchBuffer = new ArrayList<>(BATCH_SIZE);

        try (Reader reader = new InputStreamReader(file.getInputStream())) {
            var stream = CsvLoader.streamCsv(reader, timeFrame);

            stream.forEach(dto -> {
                if (DataCleaner.isValid(dto)) {
                    batchBuffer.add(dto.toEntity(symbol));
                }

                // Flush chunk to MongoDB when reaching batch size limit
                if (batchBuffer.size() >= BATCH_SIZE) {
                    executeBulkUpsert(batchBuffer);
                    batchBuffer.clear();
                }
            });

            // Flush any remaining records
            if (!batchBuffer.isEmpty()) {
                executeBulkUpsert(batchBuffer);
                batchBuffer.clear();
            }

        } catch (Exception e) {
            log.error("Error processing csv for {}", symbol, e);
        }
    }

    private void executeBulkUpsert(List<CandleStick> batchBuffer) {
        BulkOperations bulkOps = mongoTemplate.bulkOps(BulkOperations.BulkMode.UNORDERED, CandleStick.class);

        for (CandleStick candle : batchBuffer) {
            // Match document by unique composite criteria: symbol + timestamp
            Query query = new Query(
                    Criteria.where("symbol").is(candle.getSymbol())
                            .and("timestamp").is(candle.getTimestamp())
            );

            // Define fields to update/insert
            Update update = new Update()
                    .set("open", candle.getOpen())
                    .set("high", candle.getHigh())
                    .set("low", candle.getLow())
                    .set("close", candle.getClose())
                    .set("volume", candle.getVolume())
                    .set("timeFrame", candle.getTimeFrame());

            bulkOps.upsert(query, update);
        }

        bulkOps.execute();
    }

}
