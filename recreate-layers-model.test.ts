import { describe, it, expect } from "vitest";

import { tidy, zeros } from "@tensorflow/tfjs-core";
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
  it("should recreate a simple model made of a single dense layer", () => {
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

      const recreatedModel = recreateLayersModel(originalModel);

      expect(getModelSummary(recreatedModel)).toBe(
        getModelSummary(originalModel),
      );

      recreatedModel.predict(
        recreatedModel.inputs.map(({ shape }) =>
          zeros(shape.map((s) => s ?? 1)),
        ),
      );
    });
  });
});
