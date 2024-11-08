import { serialization } from "@tensorflow/tfjs-core";
import { LayersModel, layers, SymbolicTensor, model, Sequential, Shape } from "@tensorflow/tfjs-layers";

interface LayerRecreationData {
  originalLayer: layers.Layer;
  recreatedLayer: layers.Layer;
  requiresWeightsReset: boolean;
}

interface Config {
  /**
   * The new inputShapes. Each entry in {@link newInputShapes} matches one entry in `originalModel.inputLayers`.
   * Set this value to null to indicate that the input's shape should remain unchanged and the model's weights should not be reset.
   * Otherwise, set this value to the new shape.
   */
  newInputShapes?: ((number | null)[] | null)[];
}

export function recreateLayersModel(originalModel: LayersModel, { newInputShapes }: Config) {
  if (originalModel instanceof Sequential) {
    // TODO: Add support for sequential models
    throw new Error("Sequential models are not yet supported. If you need this, feel free to open an issue on https://github.com/benoitkoenig/tfjs-recreate-layers-model/issues");
  }

  if (newInputShapes && newInputShapes.length !== originalModel.inputLayers.length) {
    throw new Error("`newInputShapes` must have the same length as `originalModel.inputLayers`. Set the value to null for inputs that should remain unchanged")
  }

  const layerRecreationData: LayerRecreationData[] = [];

  const retrieveRecreatedSymbolicTensor = (
    originalSymbolicTensor: SymbolicTensor | SymbolicTensor[],
  ): SymbolicTensor | SymbolicTensor[] => {
    if (Array.isArray(originalSymbolicTensor)) {
      return originalSymbolicTensor.map(
        retrieveRecreatedSymbolicTensor,
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

  const shouldResetWeights = (originalLayer: layers.Layer) => {
    if (!newInputShapes) {
      return false;
    }

    const inputTensors = Array.isArray(originalLayer.input) ? originalLayer.input : [originalLayer.input];

    return inputTensors.some((inputTensor) => {
      const modelInputIndex = originalModel.inputLayers.indexOf(inputTensor.sourceLayer);
      
      if (modelInputIndex === -1) {
        return false;
      }

      if (newInputShapes[modelInputIndex] === null) {
        return false;
      }

      return true;
    });
  }

  for (const originalLayer of originalModel.layers) {
    if (originalModel.inputLayers.includes(originalLayer)) {
      let config = originalLayer.getConfig();

      const index = originalModel.inputLayers.indexOf(originalLayer);

      if (newInputShapes?.[index]) {
        config = { ...config, batchInputShape: newInputShapes[index] };
      }

      layerRecreationData.push({
        originalLayer,
        recreatedLayer: layers.inputLayer(config),
        requiresWeightsReset: Boolean(newInputShapes?.[index]),
      });

      continue;
    }

    if (originalLayer.inboundNodes.length !== 1) {
      throw new Error(
        "Layers with multiple inboundNodes are not supported yet.",
      );
    }

    const originalInboundNode = originalLayer.inboundNodes[0];

    const recreatedLayer =
      new (serialization.SerializationMap.getMap().classNameMap[
        originalLayer.getClassName()
      ][0])(originalLayer.getConfig()) as layers.Layer;

    recreatedLayer.apply(
      retrieveRecreatedSymbolicTensor(originalLayer.input),
      originalInboundNode.callArgs,
    );

    if (!shouldResetWeights(originalLayer)) {
      recreatedLayer.setWeights(originalLayer.getWeights());
    }

    layerRecreationData.push({
      originalLayer,
      recreatedLayer,
      requiresWeightsReset: JSON.stringify(recreatedLayer.outputShape) !== JSON.stringify(originalLayer.outputShape),
    });
  }

  return model({
    inputs: retrieveRecreatedSymbolicTensor(originalModel.inputs),
    outputs: retrieveRecreatedSymbolicTensor(originalModel.outputs),
  });
}
