import { describe, it, expect } from "vitest";
import { serialization, Tensor, tidy } from "@tensorflow/tfjs-core";
import { layers } from "@tensorflow/tfjs-layers";
import "@tensorflow/tfjs-node";

import type { LayerRecreationData } from "./types";
import retrieveReplicatedSymbolicTensor from "./retrieve-replicated-symbolic-tensor";

class SplitLayer extends layers.Layer {
  static className = "SplitLayer";

  sizes: number[];

  constructor({ sizes }: { sizes: number[] }) {
    super();
    this.sizes = sizes;
  }

  computeOutputShape(inputShape: number[]) {
    const firstShapes = inputShape.slice(0, -1);

    return this.sizes.map((size) => [...firstShapes, size]);
  }

  call(x: Tensor | Tensor[]) {
    if (Array.isArray(x)) {
      if (x.length !== 1) {
        throw new Error("Cannot split multiple tensors simulatenously");
      }

      x = x[0];
    }

    return x.split(this.sizes, 3);
  }

  override getConfig(): serialization.ConfigDict & { sizes: number[] } {
    const baseConfig = super.getConfig();

    return { ...baseConfig, sizes: this.sizes };
  }
}

serialization.registerClass(SplitLayer);

describe("Retrieve replicated symbolic tensor", () => {
  it("should return the replicated tensor corresponding to an original tensor", () => {
    tidy(() => {
      const originalLayer = layers.inputLayer({
        inputShape: [3],
      });

      const replicatedLayer = layers.inputLayer({
        inputShape: [3],
      });

      const layersRecreationData: LayerRecreationData[] = [
        {
          originalLayer,
          replicatedLayer,
          requiresWeightsReset: true,
        },
      ];

      expect(
        retrieveReplicatedSymbolicTensor(
          layersRecreationData,
          originalLayer.output,
        ),
      ).toStrictEqual(replicatedLayer.output);
    });
  });

  it("should work on a multi-output layer", () => {
    tidy(() => {
      const originalInputLayer = layers.inputLayer({
        inputShape: [4],
      });

      const replicatedInputLayer = layers.inputLayer({
        inputShape: [4],
      });

      const originalLayer = new SplitLayer({
        sizes: [2, 2],
      });

      const replicatedLayer = new SplitLayer({
        sizes: [2, 2],
      });

      originalLayer.apply(originalInputLayer.output);
      replicatedLayer.apply(replicatedInputLayer.output);

      const layersRecreationData: LayerRecreationData[] = [
        {
          originalLayer,
          replicatedLayer,
          requiresWeightsReset: false,
        },
      ];

      expect(
        retrieveReplicatedSymbolicTensor(
          layersRecreationData,
          originalLayer.output,
        ),
      ).toStrictEqual(replicatedLayer.output);
    });
  });
});
