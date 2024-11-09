import type { SymbolicTensor } from "@tensorflow/tfjs-layers";
import type { LayerRecreationData } from "./types";

export default function retrieveRecreatedSymbolicTensor(
  layerRecreationData: LayerRecreationData[],
  originalSymbolicTensor: SymbolicTensor | SymbolicTensor[],
): SymbolicTensor | SymbolicTensor[] {
  if (Array.isArray(originalSymbolicTensor)) {
    return originalSymbolicTensor.map((t) => 
      retrieveRecreatedSymbolicTensor(layerRecreationData, t),
    ) as SymbolicTensor[];
  }

  const { recreatedLayer } = layerRecreationData.find(({ originalLayer }) => originalLayer === originalSymbolicTensor.sourceLayer)!;

  if (!Array.isArray(recreatedLayer.output)) {
    return recreatedLayer.output;
  }

  throw new Error(
    "Multi-ouput layers is not yet supported in retrieveRecreatedSymbolicTensor",
  );
};
