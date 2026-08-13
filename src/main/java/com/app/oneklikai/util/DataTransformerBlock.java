package com.app.oneklikai.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.recurrent.LSTM;

public class DataTransformerBlock {

    private DataTransformerBlock() {
    }

    /**
     * Sequence encoder: [Batch, sequenceLength, inputDim] -> [Batch, 1, outputDim].
     *
     * <p>Changes that matter for prediction quality, versus a bare LSTM + Linear:
     *
     * <ul>
     *   <li><b>Input projection.</b> A per-timestep Linear + GELU lets the model mix the 23 raw
     *       features into useful combinations before the recurrence has to memorise anything.
     *   <li><b>Dropout between LSTM layers.</b> ~5k training windows against a few hundred thousand
     *       weights overfits fast; this is the cheapest regulariser that actually works on RNNs. It is
     *       automatically disabled during evaluation and inference.
     *   <li><b>Last-step + mean pooling.</b> The final hidden state alone is dominated by the last few
     *       candles. Concatenating the mean over all timesteps keeps the longer-horizon context
     *       (trend, volatility regime) available to the head.
     *   <li><b>LayerNorm before the head</b>, so the head sees a stable input distribution across the
     *       20-year span and training does not stall on a few extreme windows.
     *   <li><b>Linear output.</b> The old {@code tanh} squashed every prediction into [-1, 1] and
     *       saturated its gradient on exactly the large moves worth predicting.
     * </ul>
     */
    public static Block buildArchitecture(int inputDim,
                                          int outputDim,
                                          int embedDim,
                                          int numLayers,
                                          float dropoutRate) {
        return new SequentialBlock()

                // 1. Per-timestep feature mixing: [B, T, inputDim] -> [B, T, embedDim]
                .add(Linear.builder().setUnits(embedDim).build())
                .add(Activation::gelu)

                // 2. Recurrent encoder over the chronological window
                .add(LSTM.builder()
                        .setStateSize(embedDim)
                        .setNumLayers(numLayers)
                        .optDropRate(dropoutRate)
                        .optBatchFirst(true)
                        .build())

                // 3. Pool the sequence: final state (recent detail) + mean state (regime context)
                .add(new LambdaBlock(ndList -> {
                    NDArray sequence = ndList.getFirst();                       // [B, T, embedDim]
                    NDArray lastStep = sequence.get(new NDIndex(":, -1, :"));   // [B, embedDim]
                    NDArray meanStep = sequence.mean(new int[]{1});             // [B, embedDim]
                    return new NDList(lastStep.concat(meanStep, 1));            // [B, 2 * embedDim]
                }))

                // 4. Regression head
                .add(LayerNorm.builder().build())
                .add(Dropout.builder().optRate(dropoutRate).build())
                .add(Linear.builder().setUnits(embedDim).build())
                .add(Activation::gelu)
                .add(Linear.builder().setUnits(outputDim).build())

                // 5. [B, outputDim] -> [B, 1, outputDim] to match the label tensor
                .add(new LambdaBlock(ndList ->
                        new NDList(ndList.singletonOrThrow().expandDims(1))
                ));
    }

    public static NDArray create3DTensor(NDManager manager, float[][][] data) {
        int dim1 = data.length;
        int dim2 = data[0].length;
        int dim3 = data[0][0].length;

        // 1. Allocate flat array
        float[] flat = new float[dim1 * dim2 * dim3];
        int idx = 0;

        // 2. Flatten 3D matrix into 1D
        for (float[][] datum : data) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    flat[idx++] = datum[j][k];
                }
            }
        }

        // 3. Pass flat array along with its 3D Shape
        return manager.create(flat, new Shape(dim1, dim2, dim3));
    }

}
