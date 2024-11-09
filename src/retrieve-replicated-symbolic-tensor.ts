import type { SymbolicTensor } from "@tensorflow/tfjs-layers";
import type { LayerRecreationData } from "./types";

export default function retrieveReplicatedSymbolicTensor(
  layersRecreationData: LayerRecreationData[],
  originalSymbolicTensor: SymbolicTensor | SymbolicTensor[],
): SymbolicTensor | SymbolicTensor[] {
  if (Array.isArray(originalSymbolicTensor)) {
    return originalSymbolicTensor.map((t) =>
      retrieveReplicatedSymbolicTensor(layersRecreationData, t),
    ) as SymbolicTensor[];
  }

  const { replicatedLayer } = layersRecreationData.find(
    ({ originalLayer }) => originalLayer === originalSymbolicTensor.sourceLayer,
  )!;

  if (!Array.isArray(replicatedLayer.output)) {
    return replicatedLayer.output;
  }

  return replicatedLayer.output[originalSymbolicTensor.outputTensorIndex!];
}
