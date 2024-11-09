import { describe, it, expect } from "vitest";
import { tidy } from "@tensorflow/tfjs-core";
import { layers } from "@tensorflow/tfjs-layers";
import "@tensorflow/tfjs-node";

import type { LayerRecreationData } from "./types";
import retrieveRecreatedSymbolicTensor from "./retrieve-recreated-symbolic-tensor";

describe("Retrieve recreated symbolic tensor", () => {
  it("should return the recreated tensor corresponding to an original tensor", () => {
    tidy(() => {
      const originalLayer = layers.inputLayer({
        batchInputShape: [3],
      });

      const recreatedLayer = layers.inputLayer({
        batchInputShape: [3],
      });

      const layersRecreationData: LayerRecreationData[] = [
        {
          originalLayer,
          recreatedLayer,
          requiresWeightsReset: true,
        },
      ];

      expect(
        retrieveRecreatedSymbolicTensor(
          layersRecreationData,
          originalLayer.output,
        ),
      ).toStrictEqual(recreatedLayer.output);
    });
  });
});
