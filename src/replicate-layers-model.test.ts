import { describe, it, expect } from "vitest";

import { memory, ones, Tensor, tidy } from "@tensorflow/tfjs-core";
import {
  LayersModel,
  input,
  layers,
  model,
  SymbolicTensor,
} from "@tensorflow/tfjs-layers";
import "@tensorflow/tfjs-node";

import { replicateLayersModel } from "./replicate-layers-model";

function getModelSummary(model: LayersModel) {
  let summary = "";

  tidy(() => {
    model.summary(undefined, undefined, (str) => {
      summary += str + "\n";
    });
  });

  return summary;
}

describe("Replicate layers model", () => {
  it("should replicate a model made of a single dense layer", () => {
    tidy(() => {
      const inputLayer = input({
        shape: [3],
      });

      const output = layers
        .dense({ units: 3, kernelInitializer: "HeNormal" })
        .apply(inputLayer) as SymbolicTensor;

      const originalModel = model({
        inputs: inputLayer,
        outputs: output,
      });

      const replicatedModel = replicateLayersModel(originalModel, {});

      expect(getModelSummary(replicatedModel)).toMatchInlineSnapshot(`
        "__________________________________________________________________________________________
        Layer (type)                Input Shape               Output shape              Param #   
        ==========================================================================================
        input2 (InputLayer)         [[null,3]]                [null,3]                  0         
        __________________________________________________________________________________________
        dense_Dense2 (Dense)        [[null,3]]                [null,3]                  12        
        ==========================================================================================
        Total params: 12
        Trainable params: 12
        Non-trainable params: 0
        __________________________________________________________________________________________
        "
      `);

      const mockInput = replicatedModel.predict(
        replicatedModel.inputs.map(({ shape }) =>
          ones(shape.map((s) => s ?? 1)),
        ),
      );

      const originalPrediction = (
        originalModel.predict(mockInput) as Tensor
      ).arraySync();
      const replicatedPrediction = (
        replicatedModel.predict(mockInput) as Tensor
      ).arraySync();

      expect(replicatedPrediction).toStrictEqual(originalPrediction);
    });
  });

  it("should replicate a model and update its input", () => {
    tidy(() => {
      const inputLayer = input({
        shape: [3],
      });

      const output = layers
        .dense({ units: 3, kernelInitializer: "HeNormal" })
        .apply(inputLayer) as SymbolicTensor;

      const originalModel = model({
        inputs: inputLayer,
        outputs: output,
      });

      const replicatedModel = replicateLayersModel(originalModel, {
        newInputShapes: [[2]],
      });

      expect(getModelSummary(replicatedModel)).toMatchInlineSnapshot(`
        "__________________________________________________________________________________________
        Layer (type)                Input Shape               Output shape              Param #   
        ==========================================================================================
        input4 (InputLayer)         [[null,2]]                [null,2]                  0         
        __________________________________________________________________________________________
        dense_Dense4 (Dense)        [[null,2]]                [null,3]                  9         
        ==========================================================================================
        Total params: 9
        Trainable params: 9
        Non-trainable params: 0
        __________________________________________________________________________________________
        "
      `);

      expect(replicatedModel.inputs.map(({ shape }) => shape)).toStrictEqual([
        [null, 2],
      ]);

      const mockInput = replicatedModel.inputs.map(({ shape }) =>
        ones(shape.map((s) => s ?? 1)),
      );

      (replicatedModel.predict(mockInput) as Tensor).arraySync();
    });
  });

  it("should replicate a model and update its output", () => {
    tidy(() => {
      const inputLayer = input({
        shape: [3],
      });

      const output = layers
        .dense({ units: 3, kernelInitializer: "HeNormal" })
        .apply(inputLayer) as SymbolicTensor;

      const originalModel = model({
        inputs: inputLayer,
        outputs: output,
      });

      const replicatedModel = replicateLayersModel(originalModel, {
        newOutputFiltersOrUnits: [2],
      });

      expect(getModelSummary(replicatedModel)).toMatchInlineSnapshot(`
        "__________________________________________________________________________________________
        Layer (type)                Input Shape               Output shape              Param #   
        ==========================================================================================
        input6 (InputLayer)         [[null,3]]                [null,3]                  0         
        __________________________________________________________________________________________
        dense_Dense6 (Dense)        [[null,3]]                [null,2]                  8         
        ==========================================================================================
        Total params: 8
        Trainable params: 8
        Non-trainable params: 0
        __________________________________________________________________________________________
        "
      `);

      expect(replicatedModel.outputs.map(({ shape }) => shape)).toStrictEqual([
        [null, 2],
      ]);

      const mockInput = replicatedModel.inputs.map(({ shape }) =>
        ones(shape.map((s) => s ?? 1)),
      );

      (replicatedModel.predict(mockInput) as Tensor).arraySync();
    });
  });

  it("should dispose all superfluous tensors", () => {
    const initialNumTensors = memory().numTensors;

    const inputLayer = input({
      shape: [3],
    });

    const output = layers
      .dense({ units: 3, kernelInitializer: "HeNormal" })
      .apply(inputLayer) as SymbolicTensor;

    const originalModel = model({
      inputs: inputLayer,
      outputs: output,
    });

    const numTensorsWithOriginalModel = memory().numTensors;

    const replicatedModel = replicateLayersModel(originalModel, {});

    replicatedModel.dispose();

    expect(memory().numTensors).toBe(numTensorsWithOriginalModel);

    originalModel.dispose();

    expect(memory().numTensors).toBe(initialNumTensors);
  });
});
