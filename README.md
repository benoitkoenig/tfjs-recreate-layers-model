# tfjs-replicate-layers-model

`tfjs-replicate-layers-model` allows you to replicate an existing LayersModel. It offers options to change the model's input or output shape.

If input/output shape are updated, the weights of connected layers are reset. Intermediate weights are always preserved.

## Usage

```ts
import * as tf from "@tensorflow/tfjs";
import { replicateLayersModel } from "tfjs-replicate-layers-model";

const input = tf.input({
  shape: [3],
});

let x = input;

for (let i = 0; i < 10; i++) {
  x = tf.layers.dense({ units: 4 }).apply(x) as tf.SymbolicTensor;
}

const model = tf.model({
  inputs: input,
  outputs: x,
});

// Train the model here

const replicatedModel = replicateLayersModel(model, {
  newInputShapes: [[2]],
  newOutputFiltersOrUnits: [5],
});

model.dispose();

// `replicatedModel` takes a tensor of shape [n, 2] and outputs a tensor of shape [n, 5].
// The weights of the first and last layers are reset, but the weights of the
// intermediate layers are preserved, so training from there goes faster than re-starting from scratch.
```

## Options

### newInputShapes

The new inputShapes. Each entry in `newInputShapes` matches one entry in `originalModel.inputLayers`.
Set this value to "preserve" to indicate that the input's shape should remain unchanged and the layer's weights should not be reset.
Otherwise, set this value to the new shape.
Note that if you set this value to the same shape as previously, the input shape will remain unchanged but that will still reset the weights of layers connected to that input.

### newOutputFiltersOrUnits

The new filters (for conv layers) or units (for dense layers) to apply to the output. Each entry in `newOutputFiltersOrUnits` matches one entry in `originalModel.outputLayers`.
Set this value to null to indicate that the output's filters/units should remain unchanged and the layer's weights should not be reset.
Otherwise, set this value to the new filters/units.
Note that if you set this value to the value in the original model, the output shape will remain unchanged but that will still reset the layer's weights.

### preservedWeightsAreTrainable

If set to true, preserved weights remain trainable. Otherwise, all weights that are not reset are no longer trainable.

### verbose

Sets the verbose level.
