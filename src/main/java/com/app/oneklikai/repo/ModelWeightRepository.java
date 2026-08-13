package com.app.oneklikai.repo;

import com.app.oneklikai.model.entity.ModelWeightEntity;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ModelWeightRepository extends MongoRepository<ModelWeightEntity, String> {

    Optional<ModelWeightEntity> findBySymbol(String symbol);
}