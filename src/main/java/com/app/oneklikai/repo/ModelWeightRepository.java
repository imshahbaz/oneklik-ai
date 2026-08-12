package com.app.oneklikai.repo;

import com.app.oneklikai.model.entity.ModelWeightEntity;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ModelWeightRepository extends MongoRepository<ModelWeightEntity, String> {
}