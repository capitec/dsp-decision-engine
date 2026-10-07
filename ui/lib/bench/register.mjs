import { register } from "node:module";

// src imports are extensionless (the bundler resolves them); Node needs ".ts" spelled out.
register("data:text/javascript," + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  try { return await next(spec, ctx); } catch (e) {
    if (spec.startsWith(".")) return next(spec + ".ts", ctx);
    throw e;
  }
}`));
