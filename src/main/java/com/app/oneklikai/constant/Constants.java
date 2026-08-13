package com.app.oneklikai.constant;

import com.app.oneklikai.util.FeatureEngineer;
import com.app.oneklikai.util.FeatureScaler;

public class Constants {

    private Constants() {
    }

    // --- Data shape -------------------------------------------------------------------------------
    public static final int DAILY_SEQUENCE_LENGTH = 60;
    public static final int DAILY_FEATURE_DIM = FeatureEngineer.FEATURE_DIM;
    public static final int DAILY_TARGET_DIM = FeatureScaler.TARGET_DIM;

    /** Extra candles fetched at inference time so indicators are warmed up before the window starts. */
    public static final int DAILY_INFERENCE_LOOKBACK =
            DAILY_SEQUENCE_LENGTH + FeatureEngineer.WARMUP + 10;

    // --- Architecture ----------------------------------------------------------------------------
    public static final int DAILY_EMBED_DIM = 128;
    public static final int DAILY_NUM_LAYERS = 2;
    public static final float DAILY_DROPOUT = 0.2f;

    // --- Optimisation ----------------------------------------------------------------------------
    /**
     * The cosine schedule is annealed over exactly this many epochs, so this is a real budget rather
     * than a loose upper bound; early stopping is the safety net, not the intended exit.
     */
    public static final int DAILY_EPOCHS = 120;
    public static final float DAILY_LEARNING_RATE = 1e-3f;
    public static final int DAILY_SAMPLING_BATCH_SIZE = 64;
    public static final int DAILY_EVAL_BATCH_SIZE = 256;
    public static final float DAILY_WEIGHT_DECAY = 1e-5f;
    public static final float DAILY_GRAD_CLIP = 1.0f;
    public static final float DAILY_HUBER_DELTA = 1.0f;
    public static final int DAILY_WARMUP_STEPS = 300;
    public static final float DAILY_FINAL_LR_FRACTION = 0.02f;

    // --- Model selection -------------------------------------------------------------------------
    public static final double DAILY_VALIDATION_FRACTION = 0.10;
    public static final double DAILY_TEST_FRACTION = 0.10;
    public static final int DAILY_EARLY_STOP_PATIENCE = 30;

    /** Independent seeded runs; the one with the best validation loss is kept. */
    public static final int DAILY_TRAINING_RESTARTS = 3;
    public static final long DAILY_BASE_SEED = 20260101L;

    /** ~4 trading years: recent regimes dominate the fit without discarding older history. */
    public static final double DAILY_RECENCY_HALF_LIFE_SAMPLES = 1000.0;
}
