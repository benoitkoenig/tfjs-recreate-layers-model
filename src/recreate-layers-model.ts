import { serialization } from "@tensorflow/tfjs-core";
import { LayersModel, layers, SymbolicTensor, model, Sequential } from "@tensorflow/tfjs-layers";

interface LayerRecreationData {
  originalLayer: layers.Layer;
  recreatedLayer: layers.Layer;
  requiresWeightsReset: boolean;
}

interface Config {
  /**
   * The new inputShapes. Each entry in {@link newInputShapes} matches one entry in `originalModel.inputLayers`.
   * Set this value to null to indicate that the input's shape should remain unchanged and the layer's weights should not be reset.
   * Otherwise, set this value to the new shape.
   * Note that if you set this value to the same shape as previously, the input shape will remain unchanged but that will still reset the weights of layers connected to that input.
   */
  newInputShapes?: ((number | null)[] | null)[] | undefined;
  /**
   * The new filters (for conv layers) or units (for dense layers) to apply to the output. Each entry in {@link newOutputFiltersOrUnits} matches one entry in `originalModel.outputLayers`.
   * Set this value to null to indicate that the output's filters/units should remain unchanged and the layer's weights should not be reset.
   * Otherwise, set this value to the new filters/units.
   * Note that if you set this value to the value in the original model, the output shape will remain unchanged but that will still reset the layer's weights.
   */
  newOutputFiltersOrUnits?: (number | null)[] | undefined;
}

/**
 * Creates a new {@link LayersModel} that replicates {@link originalModel}. It is possible to change the input/output shape through the configuration parameter.
 * The recreated {@link LayersModel} has the same weight as the {@link originalModel}, except fo layers that either:
 *   - Is directly applied to an input which shape changed
 *   - Is directly applied to a layer which output shape changed
 * @param originalModel The {@link LayersModel} to replicate
 * @param config The {@link Config} for the recreated model to differ from the original model
 * @returns A {@link LayersModel}
 */
export function recreateLayersModel(originalModel: LayersModel, { newInputShapes, newOutputFiltersOrUnits }: Config) {
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

  const shouldResetWeightsBecauseOfInput = (originalLayer: layers.Layer) => {
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
    const config = { ...originalLayer.getConfig() };
    delete config["name"];

    if (originalModel.inputLayers.includes(originalLayer)) {
      const index = originalModel.inputLayers.indexOf(originalLayer);

      if (newInputShapes?.[index]) {
        config["batchInputShape"] = newInputShapes[index];
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

    let shouldResetWeightsBecauseOfOuput = false;

    if (newOutputFiltersOrUnits) {
      const indexInOutput = originalModel.outputLayers.indexOf(originalLayer);
      
      if (indexInOutput !== -1 && newOutputFiltersOrUnits[indexInOutput] !== null) {
        if (!("units" in config) && !("filters" in config)) {
          throw new Error(`Cannot update output shape of ${originalLayer.name}: no field 'units' nor 'filters' found in the layers config`);
        }

        if ("units" in config) {
          config["units"] = newOutputFiltersOrUnits[indexInOutput]!;
        }

        if ("filters" in config) {
          config["filters"] = newOutputFiltersOrUnits[indexInOutput]!;
        }

        shouldResetWeightsBecauseOfOuput = true;
      }
    }

    const recreatedLayer =
      new (serialization.SerializationMap.getMap().classNameMap[
        originalLayer.getClassName()
      ][0])(config) as layers.Layer;

    recreatedLayer.apply(
      retrieveRecreatedSymbolicTensor(originalLayer.input),
      originalInboundNode.callArgs,
    );

    if (!shouldResetWeightsBecauseOfOuput && !shouldResetWeightsBecauseOfInput(originalLayer)) {
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
