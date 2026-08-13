package com.app.oneklikai.components;

import ai.djl.Model;
import com.app.oneklikai.constant.Constants;
import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.entity.ModelWeightEntity;
import com.app.oneklikai.repo.ModelWeightRepository;
import com.app.oneklikai.util.DataTransformerBlock;
import com.app.oneklikai.util.MemoryModelSerializer;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.List;
import java.util.Optional;

@Slf4j
@Component
@RequiredArgsConstructor
public class ModelCacheManager {

    private static final Cache<String, Model> DAILY_MODEL_CACHE = Caffeine.newBuilder().maximumSize(500).build();

    private final ModelWeightRepository modelWeightRepository;

    @EventListener(ApplicationReadyEvent.class)
    public void loadGlobalModelFromDatabase() {
        log.info("Checking for pre-trained global model in database...");
        List<ModelWeightEntity> savedWeights = modelWeightRepository.findAll();
        if (CollectionUtils.isEmpty(savedWeights)) {
            log.info("No saved global model found in database...");
        }

        savedWeights.parallelStream().forEach(modelWeight -> {
            var model = Model.newInstance(modelWeight.getSymbol());
            model.setBlock(DataTransformerBlock.buildArchitecture(Constants.DAILY_FEATURE_DIM, Constants.DAILY_EMBED_DIM));

            try {
                byte[] weightBytes = modelWeight.getWeightBytes();
                MemoryModelSerializer.deserializeFromBytes(model, weightBytes, modelWeight.getSymbol());
                DAILY_MODEL_CACHE.put(model.getName(), model);
                log.info("Successfully loaded Global Model weights from DB (Updated: {}, Size: {} KB)",
                        modelWeight.getUpdatedAt(), (weightBytes.length / 1024));
            } catch (Exception e) {
                log.error("Failed to load global model weights from DB: {}", e.getMessage(), e);
            }
        });
    }

    public Optional<Model> getModel(String symbol, TimeFrame timeFrame) {
        if (TimeFrame.DAILY.equals(timeFrame)) {
            return Optional.ofNullable(DAILY_MODEL_CACHE.getIfPresent(symbol));
        }

        return Optional.empty();
    }

    public void setModel(String symbol, Model model, TimeFrame timeFrame) {
        if (TimeFrame.DAILY.equals(timeFrame)) {
            DAILY_MODEL_CACHE.put(symbol, model);
        }
    }

}
