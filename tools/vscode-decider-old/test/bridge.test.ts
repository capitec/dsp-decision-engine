import { PassThrough } from "node:stream";
import { describe, expect, it, vi } from "vitest";

const stdin = { write: vi.fn() };
const stdout = new PassThrough();
const stderr = new PassThrough();
const fd3 = new PassThrough();

vi.mock("node:child_process", () => ({
  spawn: () => ({ stdin, stdout, stderr, stdio: [null, stdout, stderr, fd3], on: vi.fn() }),
}));

const { Bridge } = await import("../src/bridge");

describe("Bridge streaming progress", () => {
  it("forwards progress lines to onProgress and only resolves on the terminal reply", async () => {
    const b = new Bridge({ python: ["python3"] });
    const progress: unknown[] = [];
    const reply = b.request("experiment", {}, (p) => progress.push(p));

    fd3.write(JSON.stringify({ id: 1, progress: { done: 1, total: 2 } }) + "\n");
    fd3.write(JSON.stringify({ id: 1, progress: { done: 2, total: 2 } }) + "\n");
    fd3.write(JSON.stringify({ id: 1, ok: true, result: { status: "completed" } }) + "\n");

    await expect(reply).resolves.toEqual({ status: "completed" });
    expect(progress).toEqual([{ done: 1, total: 2 }, { done: 2, total: 2 }]);
  });

  it("cancel sends a cancel request naming the target id", () => {
    const b = new Bridge({ python: ["python3"] });
    b.cancel(5).catch(() => undefined);
    const sent = JSON.parse(stdin.write.mock.calls.at(-1)![0] as string);
    expect(sent).toMatchObject({ cmd: "cancel", target: 5 });
  });
});
