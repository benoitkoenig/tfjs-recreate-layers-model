import typescript from "@rollup/plugin-typescript";
import { dts } from "rollup-plugin-dts";

export default [
  {
    input: "./src/index.ts",
    output: {
      format: "cjs",
      file: "dist/index.cjs",
    },
    plugins: [typescript()],
  },
  {
    input: "./src/index.ts",
    output: {
      format: "es",
      file: "dist/index.mjs",
    },
    plugins: [typescript()],
  },
  {
    input: "./src/index.ts",
    output: {
      file: "dist/index.d.ts",
    },
    plugins: [dts()],
  },
];
