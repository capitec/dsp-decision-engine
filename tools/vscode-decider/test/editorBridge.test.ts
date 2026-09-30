import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { EditorBridge, workspaceHash } from "../src/editorBridge";

// A minimal MCP-process client: connect to a window's socket, present a token, read the reply.
function client(dir: string, workspace: string, token: string, action: unknown): Promise<{ ok: boolean; result?: unknown; error?: string }> {
  return new Promise((resolve, reject) => {
    const socketPath = path.join(dir, `${workspaceHash(workspace)}.sock`);
    const conn = net.connect(socketPath, () => conn.write(JSON.stringify({ token, action }) + "\n"));
    conn.once("data", (data) => {
      resolve(JSON.parse(data.toString()));
      conn.destroy();
    });
    conn.once("error", reject);
  });
}

describe("editor bridge", () => {
  const tmp = (): string => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "decider-editor-"));
    fs.chmodSync(root, 0o700);
    return root;
  };

  const workspace = (): string => fs.mkdtempSync(path.join(os.tmpdir(), "decider-ws-"));

  it("dispatches a valid-token action and returns its result", async () => {
    const dir = tmp();
    const ws = workspace();
    const actions: unknown[] = [];
    const bridge = new EditorBridge(ws, async (a) => { actions.push(a); return { ack: true }; }, dir);
    const reply = await client(dir, ws, bridge.token, { kind: "highlight", flow: "credit", nodes: ["debt_ratio"] });
    expect(reply).toEqual({ ok: true, result: { ack: true } });
    expect(actions).toEqual([{ kind: "highlight", flow: "credit", nodes: ["debt_ratio"] }]);
    bridge.dispose();
  });

  it("rejects a wrong token", async () => {
    const dir = tmp();
    const ws = workspace();
    const bridge = new EditorBridge(ws, async () => ({ ack: true }), dir);
    const reply = await client(dir, ws, "wrong-token", { kind: "selection" });
    expect(reply).toEqual({ ok: false, error: "unauthorized" });
    bridge.dispose();
  });

  it("routes two workspaces to separate sockets", async () => {
    const dir = tmp();
    const a = workspace();
    const b = workspace();
    const gotA: unknown[] = [];
    const gotB: unknown[] = [];
    const bridgeA = new EditorBridge(a, async (x) => { gotA.push(x); return { ack: true }; }, dir);
    const bridgeB = new EditorBridge(b, async (x) => { gotB.push(x); return { ack: true }; }, dir);
    await client(dir, a, bridgeA.token, { kind: "highlight", nodes: ["one"] });
    await client(dir, b, bridgeB.token, { kind: "highlight", nodes: ["two"] });
    expect(gotA.map((x) => (x as { nodes: string[] }).nodes)).toEqual([["one"]]);
    expect(gotB.map((x) => (x as { nodes: string[] }).nodes)).toEqual([["two"]]);
    expect(workspaceHash(a)).not.toBe(workspaceHash(b));
    bridgeA.dispose();
    bridgeB.dispose();
  });

  it("removes the socket and token on dispose", async () => {
    const dir = tmp();
    const ws = workspace();
    const bridge = new EditorBridge(ws, async () => ({ ack: true }), dir);
    const sock = path.join(dir, `${workspaceHash(ws)}.sock`);
    const token = path.join(dir, `${workspaceHash(ws)}.token`);
    expect(fs.existsSync(sock)).toBe(true);
    expect(fs.existsSync(token)).toBe(true);
    bridge.dispose();
    expect(fs.existsSync(sock)).toBe(false);
    expect(fs.existsSync(token)).toBe(false);
  });
});
