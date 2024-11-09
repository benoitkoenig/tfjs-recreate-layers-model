# tfjs-recreate-layers-model

`tfjs-recreate-layers-model` allows you to replicate an existing LayersModel. It offers options to change the model's input or output shape.

If input/output shape are updated, the weights of connected layers are reset. Intermediate weights are always retained.

## Usage

```ts
import * as tf from "@tensorflow/tfjs";
import { recreateLayersModel } from "tfjs-recreate-layers-model";

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

const recreatedModel = recreateLayersModel(model, {
  newInputShapes: [[2]],
  newOutputFiltersOrUnits: [5],
});

model.dispose();

// `recreatedModel` takes a tensor of shape [n, 2] and outputs a tensor of shape [n, 5].
// The weights of the first and last layers are reset, but the weights of the
// intermediate layers are retained, so training from there goes faster than re-starting from scratch.
```
