package com.app.oneklikai.model.entity;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "model_weights")
@FieldDefaults(level = AccessLevel.PRIVATE)
public class ModelWeightEntity {

    @Id
    String name;
    byte[] weightBytes;
    int sequenceLength;
    int featureDim;
    float finalLoss;
    Instant updatedAt;
}