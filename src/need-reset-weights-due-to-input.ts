import type { layers, LayersModel } from "@tensorflow/tfjs-layers";
import type { Config } from "./recreate-layers-model";

export default function shouldResetWeightsBecauseOfInput(
  newInputShapes: Config["newInputShapes"],
  originalModel: LayersModel,
  originalLayer: layers.Layer,
) {
  if (!newInputShapes) {
    return false;
  }
  const inputTensors = Array.isArray(originalLayer.input)
    ? originalLayer.input
    : [originalLayer.input];
  return inputTensors.some((inputTensor) => {
    const modelInputIndex = originalModel.inputLayers.indexOf(
      inputTensor.sourceLayer,
    );

    if (modelInputIndex === -1) {
      return false;
    }
    if (newInputShapes[modelInputIndex] === null) {
      return false;
    }
    return true;
  });
}
