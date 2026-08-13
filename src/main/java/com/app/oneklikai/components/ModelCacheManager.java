package com.app.oneklikai.components;

import ai.djl.Model;
import com.app.oneklikai.model.TimeFrame;
import com.app.oneklikai.model.entity.ModelWeightEntity;
import com.app.oneklikai.repo.ModelWeightRepository;
import com.app.oneklikai.util.DataTransformerBlock;
import com.app.oneklikai.util.FeatureEngineer;
import com.app.oneklikai.util.MemoryModelSerializer;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Slf4j
@Component
@RequiredArgsConstructor
public class ModelCacheManager {

    /** One symbol maps to an ensemble of models whose predictions are averaged. */
    private static final Cache<String, List<Model>> DAILY_MODEL_CACHE =
            Caffeine.newBuilder().maximumSize(500).build();

    private final ModelWeightRepository modelWeightRepository;

    @EventListener(ApplicationReadyEvent.class)
    public void loadGlobalModelFromDatabase() {
        log.info("Checking for pre-trained models in database...");
        List<ModelWeightEntity> savedWeights = modelWeightRepository.findAll();
        if (CollectionUtils.isEmpty(savedWeights)) {
            log.info("No saved models found in database...");
            return;
        }

        savedWeights.parallelStream().forEach(this::loadEnsemble);
    }

    private void loadEnsemble(ModelWeightEntity weightEntity) {
        String symbol = weightEntity.getSymbol();

        // Weights are only meaningful against the feature layout they were trained on.
        if (weightEntity.getFeatureVersion() != FeatureEngineer.FEATURE_VERSION) {
            log.warn("Skipping model {}: trained with feature version {} but current version is {}. Retrain required.",
                    symbol, weightEntity.getFeatureVersion(), FeatureEngineer.FEATURE_VERSION);
            return;
        }

        List<byte[]> blobs = CollectionUtils.isEmpty(weightEntity.getEnsembleWeightBytes())
                ? List.of(weightEntity.getWeightBytes())
                : weightEntity.getEnsembleWeightBytes();

        List<Model> models = new ArrayList<>(blobs.size());
        int totalBytes = 0;
        for (int i = 0; i < blobs.size(); i++) {
            byte[] weightBytes = blobs.get(i);
            if (weightBytes == null) {
                continue;
            }

            String modelName = symbol + "-" + i;
            Model model = Model.newInstance(modelName);
            model.setBlock(DataTransformerBlock.buildArchitecture(
                    weightEntity.getFeatureDim(),
                    weightEntity.getTargetDim(),
                    weightEntity.getEmbedDim(),
                    weightEntity.getNumLayers(),
                    weightEntity.getDropout()));

            try {
                MemoryModelSerializer.deserializeFromBytes(model, weightBytes, modelName);
                models.add(model);
                totalBytes += weightBytes.length;
            } catch (Exception e) {
                model.close();
                log.error("Failed to load model weights for {} from DB: {}", modelName, e.getMessage(), e);
            }
        }

        if (models.isEmpty()) {
            log.error("No usable weights loaded for {}", symbol);
            return;
        }

        setModels(symbol, models, weightEntity.getTimeFrame());
        log.info("Loaded {} model(s) for {} (Updated: {}, Size: {} KB, Val dir-acc: {}%)",
                models.size(), symbol, weightEntity.getUpdatedAt(), totalBytes / 1024,
                String.format("%.2f", weightEntity.getValidationDirectionalAccuracyPercent()));
    }

    public Optional<List<Model>> getModels(String symbol, TimeFrame timeFrame) {
        if (TimeFrame.DAILY.equals(timeFrame)) {
            return Optional.ofNullable(DAILY_MODEL_CACHE.getIfPresent(symbol.toUpperCase()));
        }

        return Optional.empty();
    }

    public void setModels(String symbol, List<Model> models, TimeFrame timeFrame) {
        if (TimeFrame.DAILY.equals(timeFrame)) {
            // Superseded models are intentionally left open: a concurrent prediction may still hold
            // the previous list, and closing it under an in-flight predictor crashes the native call.
            DAILY_MODEL_CACHE.put(symbol.toUpperCase(), List.copyOf(models));
        }
    }

}
