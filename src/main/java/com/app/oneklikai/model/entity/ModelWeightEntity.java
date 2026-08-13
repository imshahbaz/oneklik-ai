package com.app.oneklikai.model.entity;

import com.app.oneklikai.model.TimeFrame;
import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "model_weights")
@FieldDefaults(level = AccessLevel.PRIVATE)
public class ModelWeightEntity {

    @Id
    String id;
    String symbol;
    TimeFrame timeFrame;

    /** Weights of the single best run, kept for tooling that expects one blob. */
    byte[] weightBytes;

    /** Every retained run, averaged at inference time; the first entry is the best run. */
    List<byte[]> ensembleWeightBytes;

    int sequenceLength;
    int featureDim;
    int targetDim;
    int embedDim;
    int numLayers;
    float dropout;

    /** Feature layout the weights were trained against; stale versions are not loaded. */
    int featureVersion;

    int trainSamples;
    double validationLoss;
    double validationDirectionalAccuracyPercent;
    double testDirectionalAccuracyPercent;
    Instant updatedAt;
}
