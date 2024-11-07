import typescript from "@rollup/plugin-typescript";
import { dts } from "rollup-plugin-dts";

export default {
  input: "./index.ts",
  output: {
    dir: "dist",
    format: "cjs"
  },
  plugins: [typescript(), dts()]
};
