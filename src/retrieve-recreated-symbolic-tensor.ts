import type { SymbolicTensor } from "@tensorflow/tfjs-layers";
import type { LayerRecreationData } from "./types";

export default function retrieveRecreatedSymbolicTensor(
  layersRecreationData: LayerRecreationData[],
  originalSymbolicTensor: SymbolicTensor | SymbolicTensor[],
): SymbolicTensor | SymbolicTensor[] {
  if (Array.isArray(originalSymbolicTensor)) {
    return originalSymbolicTensor.map((t) =>
      retrieveRecreatedSymbolicTensor(layersRecreationData, t),
    ) as SymbolicTensor[];
  }

  const { recreatedLayer } = layersRecreationData.find(
    ({ originalLayer }) => originalLayer === originalSymbolicTensor.sourceLayer,
  )!;

  if (!Array.isArray(recreatedLayer.output)) {
    return recreatedLayer.output;
  }

  throw new Error(
    "Multi-ouput layers is not yet supported in retrieveRecreatedSymbolicTensor",
  );
}
