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
import ai.djl.nn.recurrent.LSTM;
import ai.djl.nn.transformer.TransformerEncoderBlock;

public class DataTransformerBlock {

    public static Block buildArchitecture(int featureDim, int hiddenDim) {
        return new SequentialBlock()

                // 1. LSTM Layer: Processes the 60-day chronological sequence
                // Input shape: [Batch, 60 days, 5 features]
                .add(LSTM.builder()
                        .setStateSize(hiddenDim) // e.g., 64 dimensions of "memory"
                        .setNumLayers(2)         // 2 stacked LSTM layers for deeper learning
                        .optBatchFirst(true)     // Crucial: tells DJL our batch size is the first dimension
                        .build())

                // 2. Extract Final Step: We only care about the LSTM's state after reading Day 60
                .add(new LambdaBlock(ndList -> {
                    // LSTM returns an NDList. Index 0 contains the full sequence output: [Batch, 60, HiddenDim]
                    // We use NDIndex slicing to grab the last time step (axis 1, index -1)
                    return new NDList(ndList.getFirst().get(new NDIndex(":, -1, :")));
                }))

                // 3. Dense Projection: Map the 64-dim hidden state back to 5 predicted values
                // Output shape: [Batch, 5]
                .add(Linear.builder().setUnits(featureDim).build())
                .add(Activation::tanh)

                // 4. Reshape to match our Target Tensor
                // Expands [Batch, 5] to [Batch, 1, 5] so it matches our Day 61 Ground Truth array
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
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    flat[idx++] = data[i][j][k];
                }
            }
        }

        // 3. Pass flat array along with its 3D Shape
        return manager.create(flat, new Shape(dim1, dim2, dim3));
    }

}