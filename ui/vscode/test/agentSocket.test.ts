import { mkdtempSync, mkdirSync, readFileSync, realpathSync, existsSync, statSync, writeFileSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { folderHash, listen } from "../src/agentSocket";

const stops: (() => void)[] = [];
afterEach(() => void stops.splice(0).forEach((s) => s()));

const tmp = () => mkdtempSync(path.join(process.env.TMPDIR ?? tmpdir(), "agent-"));

function call(sock: string, req: object): Promise<any> {
  return new Promise((resolve, reject) => {
    const c = net.connect(sock);
    let buf = "";
    c.on("data", (d) => (buf += d));
    c.on("end", () => resolve(JSON.parse(buf)));
    c.on("error", reject);
    c.write(JSON.stringify(req) + "\n");
  });
}

async function serveFolder(handle: (a: any) => Promise<unknown> = async (a) => ({ echo: a })) {
  const folder = tmp();
  const dir = path.join(tmp(), "editor");
  stops.push(await listen(folder, handle, dir));
  const name = folderHash(folder);
  return { folder, dir, name, sock: path.join(dir, `${name}.sock`), token: readFileSync(path.join(dir, `${name}.token`), "utf8") };
}

describe("listen", () => {
  it("answers an authorised action", async () => {
    const w = await serveFolder();
    expect(await call(w.sock, { token: w.token, action: { kind: "selection" } })).toEqual({ ok: true, result: { echo: { kind: "selection" } } });
  });

  it("rejects a wrong token without running the action", async () => {
    let ran = false;
    const w = await serveFolder(async () => void (ran = true));
    expect(await call(w.sock, { token: "nope", action: { kind: "x" } })).toEqual({ ok: false, error: "unauthorized" });
    expect(ran).toBe(false);
  });

  it("returns a handler error as an error reply", async () => {
    const w = await serveFolder(async () => { throw new Error("no flow"); });
    expect(await call(w.sock, { token: w.token, action: {} })).toEqual({ ok: false, error: "no flow" });
  });

  it("keeps the files private and removes them on stop", async () => {
    const w = await serveFolder();
    expect(statSync(w.dir).mode & 0o777).toBe(0o700);
    expect(statSync(path.join(w.dir, `${w.name}.token`)).mode & 0o777).toBe(0o600);
    expect(readFileSync(path.join(w.dir, `${w.name}.folder`), "utf8")).toBe(realpathSync(w.folder));
    stops.pop()!();
    expect(existsSync(w.sock)).toBe(false);
    expect(existsSync(path.join(w.dir, `${w.name}.token`))).toBe(false);
  });

  it("replaces a socket a crashed window left behind", async () => {
    const folder = tmp();
    const dir = path.join(tmp(), "editor");
    mkdirSync(dir);
    const name = folderHash(folder);
    // A crash leaves the socket file with nobody listening.
    writeFileSync(path.join(dir, `${name}.sock`), "");
    stops.push(await listen(folder, async () => "fresh", dir));
    const token = readFileSync(path.join(dir, `${name}.token`), "utf8");
    expect((await call(path.join(dir, `${name}.sock`), { token, action: {} })).result).toBe("fresh");
  });

  it("leaves a live window's socket alone", async () => {
    const w = await serveFolder(async () => "first");
    const stop = await listen(w.folder, async () => "second", w.dir);
    stop();
    const token = readFileSync(path.join(w.dir, `${w.name}.token`), "utf8");
    expect((await call(w.sock, { token, action: {} })).result).toBe("first");
  });
});
