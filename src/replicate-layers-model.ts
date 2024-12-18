import { serialization } from "@tensorflow/tfjs-core";
import {
  LayersModel,
  layers,
  model,
  Sequential,
  Shape,
} from "@tensorflow/tfjs-layers";
import type { LayerRecreationData } from "./types";
import retrieveReplicatedSymbolicTensor from "./retrieve-replicated-symbolic-tensor";
import shouldResetWeightsBecauseOfInput from "./need-reset-weights-due-to-input";

export interface Config {
  /**
   * The new inputShapes. Each entry in {@link newInputShapes} matches one entry in `originalModel.inputLayers`.
   * Set this value to "preserve" to indicate that the input's shape should remain unchanged and the layer's weights should not be reset.
   * Otherwise, set this value to the new shape.
   * Note that if you set this value to the same shape as previously, the input shape will remain unchanged but that will still reset the weights of layers connected to that input.
   */
  newInputShapes?: (Shape | "preserve")[] | undefined;
  /**
   * The new filters (for conv layers) or units (for dense layers) to apply to the output. Each entry in {@link newOutputFiltersOrUnits} matches one entry in `originalModel.outputLayers`.
   * Set this value to null to indicate that the output's filters/units should remain unchanged and the layer's weights should not be reset.
   * Otherwise, set this value to the new filters/units.
   * Note that if you set this value to the value in the original model, the output shape will remain unchanged but that will still reset the layer's weights.
   */
  newOutputFiltersOrUnits?: Shape | undefined;
  /**
   * If set to true, preserved weights remain trainable. Otherwise, all weights that are not reset are no longer trainable.
   */
  preservedWeightsAreTrainable?: boolean;
  verbose?: boolean | undefined;
}

/**
 * Creates a new {@link LayersModel} that replicates {@link originalModel}. It is possible to change the input/output shape through the configuration parameter.
 * The replicated {@link LayersModel} has the same weight as the {@link originalModel}, except fo layers that either:
 *   - Is directly applied to an input which shape changed
 *   - Is directly applied to a layer which output shape changed
 * @param originalModel The {@link LayersModel} to replicate
 * @param config The {@link Config} for the replicated model to differ from the original model
 * @returns A {@link LayersModel}
 */
export function replicateLayersModel(
  originalModel: LayersModel,
  {
    newInputShapes,
    newOutputFiltersOrUnits,
    preservedWeightsAreTrainable,
    verbose,
  }: Config,
) {
  if (originalModel instanceof Sequential) {
    // TODO: Add support for sequential models
    throw new Error(
      "Sequential models are not yet supported. If you need this, feel free to open an issue on https://github.com/benoitkoenig/tfjs-replicate-layers-model/issues",
    );
  }

  if (
    newInputShapes &&
    newInputShapes.length !== originalModel.inputLayers.length
  ) {
    throw new Error(
      "`newInputShapes` must have the same length as `originalModel.inputLayers`. Set the value to null for inputs that should remain unchanged",
    );
  }

  const layersRecreationData: LayerRecreationData[] = [];
  const inputsWithResetWeights: string[] = [];

  for (const originalLayer of originalModel.layers) {
    const config = { ...originalLayer.getConfig() };
    delete config["name"];

    if (originalModel.inputLayers.includes(originalLayer)) {
      const index = originalModel.inputLayers.indexOf(originalLayer);

      if (newInputShapes && newInputShapes[index] !== "preserve") {
        delete config["batchInputShape"];
        config["inputShape"] = newInputShapes[index];
      }

      layersRecreationData.push({
        originalLayer,
        replicatedLayer: layers.inputLayer(config),
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

      if (
        indexInOutput !== -1 &&
        newOutputFiltersOrUnits[indexInOutput] !== null
      ) {
        if (!("units" in config) && !("filters" in config)) {
          throw new Error(
            `Cannot update output shape of ${originalLayer.name}: no field 'units' nor 'filters' found in the layers config`,
          );
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

    const replicatedLayer =
      new (serialization.SerializationMap.getMap().classNameMap[
        originalLayer.getClassName()
      ][0])(config) as layers.Layer;

    replicatedLayer.apply(
      retrieveReplicatedSymbolicTensor(
        layersRecreationData,
        originalLayer.input,
      ),
      originalInboundNode.callArgs,
    );

    if (
      !shouldResetWeightsBecauseOfOuput &&
      !shouldResetWeightsBecauseOfInput(
        newInputShapes,
        originalModel,
        originalLayer,
      )
    ) {
      replicatedLayer.setWeights(originalLayer.getWeights());
    } else {
      inputsWithResetWeights.push(originalLayer.name);

      if (!preservedWeightsAreTrainable) {
        replicatedLayer.trainable = false;
      }
    }

    layersRecreationData.push({
      originalLayer,
      replicatedLayer,
      requiresWeightsReset:
        JSON.stringify(replicatedLayer.outputShape) !==
        JSON.stringify(originalLayer.outputShape),
    });
  }

  if (verbose) {
    console.log(
      `The weights of layers ${inputsWithResetWeights.join(", ")} in model ${originalModel.name} are reset`,
    );
  }

  return model({
    inputs: retrieveReplicatedSymbolicTensor(
      layersRecreationData,
      originalModel.inputs,
    ),
    outputs: retrieveReplicatedSymbolicTensor(
      layersRecreationData,
      originalModel.outputs,
    ),
  });
}
