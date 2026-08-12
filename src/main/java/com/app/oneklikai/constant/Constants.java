package com.app.oneklikai.constant;

import ai.djl.Model;

public class Constants {
    public static final String DAILY_MODEL_NAME = "one-klik-daily";
    public static Model DAILY_MODEL = Model.newInstance(DAILY_MODEL_NAME);
    public static final int DAILY_SEQUENCE_LENGTH = 60;
    public static final int DAILY_EPOCHS = 60;
    public static final float DAILY_LEARNING_RATE = 0.0005f;
    public static final int DAILY_FEATURE_DIM = 5;
    public static final int DAILY_EMBED_DIM = 64;
}
