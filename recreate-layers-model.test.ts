import { describe, it, expect } from "vitest";

import { ones, Tensor, tensor1d, tensor2d, tidy } from "@tensorflow/tfjs-core";
import { LayersModel, input, layers, model, SymbolicTensor } from "@tensorflow/tfjs-layers";
import "@tensorflow/tfjs-node";

import { recreateLayersModel } from "./recreate-layers-model";

function getModelSummary(model: LayersModel) {
  let summary = "";

  tidy(() => {
    model.summary(undefined, undefined, (str) => {
      summary += str + "\n";
    });
  });

  return summary;
}

describe("Recreate layers model", () => {
  it("should recreate a model made of a single dense layer", () => {
    tidy(() => {
      const inputLayer = input({
        shape: [3],
      });

      const output = layers
        .dense({ units: 3 })
        .apply(inputLayer) as SymbolicTensor;

      const originalModel = model({
        inputs: inputLayer,
        outputs: output,
      });

      originalModel.trainableWeights[0].write(tensor2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
      originalModel.trainableWeights[1].write(tensor1d([10, 20, 30]));

      const recreatedModel = recreateLayersModel(originalModel);

      expect(getModelSummary(recreatedModel)).toBe(
        getModelSummary(originalModel),
      );

      const mockInput = recreatedModel.predict(
        recreatedModel.inputs.map(({ shape }) =>
          ones(shape.map((s) => s ?? 1)),
        ),
      )

      const originalPrediction = (originalModel.predict(mockInput) as Tensor).arraySync();
      const recreatedPrediction = (recreatedModel.predict(mockInput) as Tensor).arraySync();

      expect(originalPrediction).toMatchInlineSnapshot(`
        [
          [
            508,
            623,
            738,
          ],
        ]
      `);
      expect(recreatedPrediction).toStrictEqual(originalPrediction);
    });
  });
});
