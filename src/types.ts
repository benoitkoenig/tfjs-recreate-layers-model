import type { layers } from "@tensorflow/tfjs-layers";

export interface LayerRecreationData {
    originalLayer: layers.Layer;
    recreatedLayer: layers.Layer;
    requiresWeightsReset: boolean;
  }
  