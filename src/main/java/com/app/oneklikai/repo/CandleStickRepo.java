package com.app.oneklikai.repo;

import com.app.oneklikai.model.entity.CandleStick;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Stream;

@Repository
public interface CandleStickRepo extends MongoRepository<CandleStick, String> {

    Stream<CandleStick> findBySymbolOrderByTimestampAsc(String symbol);

    List<CandleStick> findTop60BySymbolOrderByTimestampDesc(String symbol);
}
