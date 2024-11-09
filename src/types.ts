import type { layers } from "@tensorflow/tfjs-layers";

export interface LayerRecreationData {
  originalLayer: layers.Layer;
  replicatedLayer: layers.Layer;
  requiresWeightsReset: boolean;
}
