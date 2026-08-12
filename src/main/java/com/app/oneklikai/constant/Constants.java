package com.app.oneklikai.constant;

import ai.djl.Model;

public class Constants {
    public static final Model DAILY_MODEL = Model.newInstance("one-klik-daily");
    public static final int DAILY_SEQUENCE_LENGTH = 60;
    public static final int DAILY_EPOCHS = 60;
    public static final float DAILY_LEARNING_RATE = 0.0005f;
}
