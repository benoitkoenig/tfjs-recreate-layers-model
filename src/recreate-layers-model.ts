import { serialization } from "@tensorflow/tfjs-core";
import { LayersModel, layers, SymbolicTensor, model, Sequential } from "@tensorflow/tfjs-layers";

export function recreateLayersModel(originalModel: LayersModel) {
  if (originalModel instanceof Sequential) {
    // TODO: Add support for sequential models
    throw new Error("Sequential models are not yet supported. If you need this, feel free to open an issue on https://github.com/benoitkoenig/tfjs-recreate-layers-model/issues");
  }

  const recreatedLayers: layers.Layer[] = [];

  const retrieveRecreatedSymbolicTensor = (
    originalSymbolicTensor: SymbolicTensor | SymbolicTensor[],
  ): SymbolicTensor | SymbolicTensor[] => {
    if (Array.isArray(originalSymbolicTensor)) {
      return originalSymbolicTensor.map(
        retrieveRecreatedSymbolicTensor,
      ) as SymbolicTensor[];
    }

    const layerIndex = originalModel.layers.indexOf(
      originalSymbolicTensor.sourceLayer,
    );
    const recreatedLayer = recreatedLayers[layerIndex];

    if (!Array.isArray(recreatedLayer.output)) {
      return recreatedLayer.output;
    }

    throw new Error(
      "Multi-ouput layers is not yet supported in retrieveRecreatedSymbolicTensor",
    );
  };

  for (const originalLayer of originalModel.layers) {
    if (originalModel.inputLayers.includes(originalLayer)) {
      recreatedLayers.push(layers.inputLayer(originalLayer.getConfig()));

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

    recreatedLayer.setWeights(originalLayer.getWeights());

    recreatedLayers.push(recreatedLayer);
  }

  return model({
    inputs: retrieveRecreatedSymbolicTensor(originalModel.inputs),
    outputs: retrieveRecreatedSymbolicTensor(originalModel.outputs),
  });
}
