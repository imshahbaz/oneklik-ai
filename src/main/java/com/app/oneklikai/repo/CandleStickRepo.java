package com.app.oneklikai.repo;

import com.app.oneklikai.model.entity.CandleStick;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Stream;

@Repository
public interface CandleStickRepo extends MongoRepository<CandleStick, String> {

    Stream<CandleStick> findBySymbolOrderByTimestampAsc(String symbol);

    /** Most recent candles first; the caller reverses them into chronological order. */
    List<CandleStick> findBySymbolOrderByTimestampDesc(String symbol, Pageable pageable);
}
