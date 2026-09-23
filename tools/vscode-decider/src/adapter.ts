import * as net from "node:net";
import * as path from "node:path";
import {
  Breakpoint,
  Event,
  Handles,
  InitializedEvent,
  LoggingDebugSession,
  OutputEvent,
  Scope,
  Source,
  StackFrame,
  StoppedEvent,
  TerminatedEvent,
  Thread,
  Variable,
} from "@vscode/debugadapter";
import type { DebugProtocol } from "@vscode/debugprotocol";
import { Bridge } from "./bridge";
import {
  callNodes,
  lastSegment,
  walk,
  type Checkpoint,
  type ColumnSummary,
  type DescribeResult,
  type IRNodeJson,
  type Lineage,
  type Status,
} from "./protocol";

export interface AdapterOptions {
  python: string[];
  debugpyLibs?: string;
}

export interface LaunchArgs extends DebugProtocol.LaunchRequestArguments {
  program: string;
  pipeline?: string;
  data?: unknown;
  stopOnEntry?: boolean;
  cwd?: string;
}

type VarRef =
  | { kind: "scope"; names: string[] | "all" }
  | { kind: "column"; name: string }
  | { kind: "values"; name: string }
  | { kind: "versions"; name: string }
  | { kind: "lineage"; entry: Lineage };

const THREAD = 1;

/**
 * Maps the Debug Adapter Protocol onto a decider session: one thread, a
 * stack made of the current node's ancestors, scopes for the node's inputs,
 * outputs and the whole state, `setVariable` as `set`, and `restartFrame` as
 * `rewind`.
 */
export class DeciderDebugSession extends LoggingDebugSession {
  private bridge?: Bridge;
  private describe?: DescribeResult;
  private nodes = new Map<string, IRNodeJson>();
  private parents = new Map<string, string>();
  private current: Checkpoint | null = null;
  private finished = false;
  private finishedPaths: string[] = [];
  private handles = new Handles<VarRef>();
  private frameIds = new Map<number, string>();
  private stateCache?: Promise<ColumnSummary[]>;
  private sourceBps = new Map<string, string[]>();
  private fnBps: string[] = [];
  private applied = new Set<string>();
  private configured!: () => void;
  private configuredPromise = new Promise<void>((r) => (this.configured = r));
  private launchArgs?: LaunchArgs;
  private debugpyPort?: number;

  private opts: AdapterOptions;

  constructor(opts?: AdapterOptions | boolean) {
    super();
    // DebugSession.run passes a legacy boolean; the stdio entry then configures from the environment.
    this.opts = typeof opts === "object" ? opts : optionsFromEnv();
    this.setDebuggerLinesStartAt1(true);
    this.setDebuggerColumnsStartAt1(true);
  }

  protected initializeRequest(response: DebugProtocol.InitializeResponse): void {
    response.body = {
      supportsConfigurationDoneRequest: true,
      supportsFunctionBreakpoints: true,
      supportsSetVariable: true,
      supportsRestartFrame: true,
      supportsRestartRequest: true,
      supportsEvaluateForHovers: true,
      supportsTerminateRequest: true,
    };
    this.sendResponse(response);
  }

  protected configurationDoneRequest(response: DebugProtocol.ConfigurationDoneResponse): void {
    this.sendResponse(response);
    this.configured();
  }

  protected async launchRequest(response: DebugProtocol.LaunchResponse, args: LaunchArgs): Promise<void> {
    this.launchArgs = args;
    try {
      this.debugpyPort = this.opts.debugpyLibs ? await freePort() : undefined;
      this.bridge = new Bridge({
        python: this.opts.python,
        cwd: args.cwd ?? path.dirname(args.program),
        pythonPath: this.opts.debugpyLibs ? [this.opts.debugpyLibs] : undefined,
        debugpyPort: this.debugpyPort,
        onOutput: (text, category) => this.sendEvent(new OutputEvent(text, category)),
      });
      this.bridge.exited.then(() => {
        if (!this.finished) this.sendEvent(new TerminatedEvent());
      });
      this.describe = await this.bridge.request<DescribeResult>("describe", { file: args.program, pipeline: args.pipeline });
      walk(this.describe.ir, (n, parent) => {
        this.nodes.set(n.path, n);
        if (parent) this.parents.set(n.path, parent.path);
      });
      this.sendResponse(response);
      // Breakpoints arrive after this event; launch continues once configurationDone lands.
      this.sendEvent(new InitializedEvent());
      await this.configuredPromise;
      const status = await this.bridge.request<Status>("start", {
        file: args.program,
        pipeline: args.pipeline,
        data: args.data ?? null,
        breakpoints: this.desiredBreakpoints(),
      });
      this.applied = new Set(this.desiredBreakpoints());
      this.apply(status, false);
      this.sendEvent(new OutputEvent(`decider: running ${this.describe.pipeline} from ${args.program}\n`, "console"));
      await this.run(args.stopOnEntry === false ? "resume" : "step_into", {}, args.stopOnEntry === false ? undefined : "entry");
    } catch (e) {
      this.sendErrorResponse(response, 1, `decider: ${(e as Error).message}`);
      this.sendEvent(new TerminatedEvent());
    }
  }

  // ---------------------------------------------------------------- breakpoints

  private desiredBreakpoints(): string[] {
    return [...new Set([...this.fnBps, ...[...this.sourceBps.values()].flat()])];
  }

  private async syncBreakpoints() {
    if (!this.bridge) return;
    const desired = new Set(this.desiredBreakpoints());
    for (const p of desired) if (!this.applied.has(p)) await this.bridge.request("break_at", { path: p });
    for (const p of this.applied) if (!desired.has(p)) await this.bridge.request("clear_break", { path: p });
    this.applied = desired;
  }

  protected async setBreakPointsRequest(
    response: DebugProtocol.SetBreakpointsResponse,
    args: DebugProtocol.SetBreakpointsArguments,
  ): Promise<void> {
    const file = args.source.path ?? "";
    const inFile = [...this.nodes.values()].filter((n) => n.file && samePath(n.file, file) && n.line !== null);
    const paths: string[] = [];
    response.body = {
      breakpoints: (args.breakpoints ?? []).map((bp) => {
        // The node whose definition starts closest above the requested line.
        const node = inFile.filter((n) => n.line! <= bp.line).sort((a, b) => b.line! - a.line!)[0];
        if (!node) return new Breakpoint(false);
        paths.push(node.path);
        const b = new Breakpoint(true, node.line!, undefined, new Source(path.basename(file), file));
        (b as DebugProtocol.Breakpoint).message = node.path;
        return b;
      }),
    };
    this.sourceBps.set(file, paths);
    if (this.current || this.finished) await this.syncBreakpoints();
    this.sendResponse(response);
  }

  protected async setFunctionBreakPointsRequest(
    response: DebugProtocol.SetFunctionBreakpointsResponse,
    args: DebugProtocol.SetFunctionBreakpointsArguments,
  ): Promise<void> {
    this.fnBps = args.breakpoints.map((b) => b.name);
    response.body = { breakpoints: this.fnBps.map(() => new Breakpoint(true)) };
    if (this.current || this.finished) await this.syncBreakpoints();
    this.sendResponse(response);
  }

  // ---------------------------------------------------------------- running

  private async run(cmd: string, args: Record<string, unknown> = {}, reason?: string): Promise<void> {
    const status = await this.bridge!.request<Status>(cmd, args);
    this.apply(status, true, reason);
  }

  private apply(status: Status, stop: boolean, reason?: string) {
    let paused = reason;
    for (const ev of status.events) {
      switch (ev.event) {
        case "NodeFinished":
          if (ev.outputs?.length) {
            this.finishedPaths.push(ev.path!);
            this.sendEvent(new OutputEvent(`${ev.path}  ${ev.outputs.map((o) => `${o.name}=${preview(o)}`).join("  ")}\n`, "stdout"));
          }
          break;
        case "Overridden":
          this.sendEvent(new OutputEvent(`set ${ev.name} (${ev.producer})\n`, "console"));
          break;
        case "RunFinished":
          this.sendEvent(new OutputEvent(`run finished; columns: ${(ev.columns as string[]).join(", ")}\n`, "console"));
          break;
        case "Paused":
          paused ??= ev.reason;
          break;
      }
    }
    this.current = status.current;
    this.finished = status.finished;
    this.stateCache = undefined;
    this.handles.reset();
    this.sendEvent(new Event("decider.status", { current: this.current, finished: this.finished, finishedPaths: this.finishedPaths }));
    if (!stop) return;
    if (this.finished) this.sendEvent(new TerminatedEvent());
    else this.sendEvent(new StoppedEvent(paused ?? "step", THREAD));
  }

  protected async nextRequest(response: DebugProtocol.NextResponse): Promise<void> {
    this.sendResponse(response);
    await this.run("step");
  }

  protected async stepInRequest(response: DebugProtocol.StepInResponse): Promise<void> {
    this.sendResponse(response);
    await this.run("step_into");
  }

  protected async stepOutRequest(response: DebugProtocol.StepOutResponse): Promise<void> {
    // ponytail: Session has no step-out; step-over is the nearest. Add step_out to Session if people miss it.
    this.sendResponse(response);
    await this.run("step");
  }

  protected async continueRequest(response: DebugProtocol.ContinueResponse): Promise<void> {
    this.sendResponse(response);
    await this.run("resume");
  }

  protected pauseRequest(response: DebugProtocol.PauseResponse): void {
    this.bridge?.request("pause").catch(() => undefined);
    this.sendResponse(response);
  }

  protected async restartFrameRequest(
    response: DebugProtocol.RestartFrameResponse,
    args: DebugProtocol.RestartFrameArguments,
  ): Promise<void> {
    this.sendResponse(response);
    await this.run("rewind", { path: this.frameIds.get(args.frameId) ?? "" });
  }

  protected async restartRequest(response: DebugProtocol.RestartResponse): Promise<void> {
    this.sendResponse(response);
    this.finishedPaths = [];
    const status = await this.bridge!.request<Status>("start", {
      file: this.launchArgs!.program,
      pipeline: this.launchArgs!.pipeline,
      data: this.launchArgs!.data ?? null,
      breakpoints: this.desiredBreakpoints(),
    });
    this.apply(status, false);
    await this.run("step_into", {}, "entry");
  }

  protected async disconnectRequest(response: DebugProtocol.DisconnectResponse): Promise<void> {
    this.finished = true;
    await this.bridge?.dispose();
    this.sendResponse(response);
  }

  protected async terminateRequest(response: DebugProtocol.TerminateResponse): Promise<void> {
    this.finished = true;
    await this.bridge?.dispose();
    this.sendResponse(response);
    this.sendEvent(new TerminatedEvent());
  }

  // ---------------------------------------------------------------- inspection

  protected threadsRequest(response: DebugProtocol.ThreadsResponse): void {
    response.body = { threads: [new Thread(THREAD, "pipeline")] };
    this.sendResponse(response);
  }

  protected stackTraceRequest(response: DebugProtocol.StackTraceResponse): void {
    this.frameIds.clear();
    const frames: StackFrame[] = [];
    let p: string | undefined = this.current?.path;
    while (p !== undefined) {
      const node = this.nodes.get(p);
      const id = frames.length + 1;
      this.frameIds.set(id, p);
      const src = node?.file ? new Source(path.basename(node.file), node.file) : undefined;
      frames.push(new StackFrame(id, `${lastSegment(p)}  [${node?.kind ?? "?"}]`, src, node?.line ?? 0, 1));
      p = this.parents.get(p);
    }
    response.body = { stackFrames: frames, totalFrames: frames.length };
    this.sendResponse(response);
  }

  protected scopesRequest(response: DebugProtocol.ScopesResponse, args: DebugProtocol.ScopesArguments): void {
    const node = this.nodes.get(this.frameIds.get(args.frameId) ?? "");
    const scopes: Scope[] = [];
    if (node?.kind === "call") {
      scopes.push(new Scope("Inputs", this.handles.create({ kind: "scope", names: node.inputs }), false));
      scopes.push(new Scope("Outputs", this.handles.create({ kind: "scope", names: node.outputs }), false));
    }
    scopes.push(new Scope("State", this.handles.create({ kind: "scope", names: "all" }), true));
    response.body = { scopes };
    this.sendResponse(response);
  }

  private state(): Promise<ColumnSummary[]> {
    this.stateCache ??= this.bridge!.request<{ columns: ColumnSummary[] }>("state").then((r) => r.columns);
    return this.stateCache;
  }

  protected async variablesRequest(
    response: DebugProtocol.VariablesResponse,
    args: DebugProtocol.VariablesArguments,
  ): Promise<void> {
    const ref = this.handles.get(args.variablesReference);
    let variables: Variable[] = [];
    try {
      variables = await this.variablesFor(ref);
    } catch (e) {
      variables = [new Variable("error", (e as Error).message)];
    }
    response.body = { variables };
    this.sendResponse(response);
  }

  private async variablesFor(ref: VarRef): Promise<Variable[]> {
    switch (ref.kind) {
      case "scope": {
        const cols = await this.state();
        const byName = new Map(cols.map((c) => [c.name, c]));
        const names = ref.names === "all" ? cols.map((c) => c.name) : ref.names;
        return names.map((n) => {
          const c = byName.get(n);
          if (!c) return new Variable(n, "<not yet computed>");
          const v: DebugProtocol.Variable = new Variable(n, preview(c), this.handles.create({ kind: "column", name: n }));
          v.type = c.dtype;
          v.evaluateName = n;
          return v as Variable;
        });
      }
      case "column": {
        const c = (await this.state()).find((x) => x.name === ref.name)!;
        return [
          new Variable("values", `${c.rows} rows`, this.handles.create({ kind: "values", name: ref.name })),
          new Variable("producer", c.producer),
          new Variable("nulls", String(c.nulls)),
          new Variable("versions", String(c.versions), this.handles.create({ kind: "versions", name: ref.name })),
          new Variable("lineage", "static, before the current node", this.handles.create({ kind: "lineage", entry: await this.lineage(ref.name) })),
        ];
      }
      case "values": {
        const col = await this.bridge!.request<{ values: unknown[] }>("column", { name: ref.name });
        return col.values.map((v, i) => new Variable(`[${i}]`, JSON.stringify(v)));
      }
      case "versions": {
        const col = await this.bridge!.request<{ versions: { producer: string; values: unknown[] }[] }>("column", { name: ref.name });
        return col.versions.map((v) => new Variable(v.producer, JSON.stringify(v.values.slice(0, 5))));
      }
      case "lineage":
        return ref.entry.inputs.map(
          (e) => new Variable(e.name, e.producer ?? "<input column>", e.inputs.length ? this.handles.create({ kind: "lineage", entry: e }) : 0),
        );
    }
  }

  private lineage(name: string): Promise<Lineage> {
    return this.bridge!.request<Lineage>("lineage", { name, path: this.current?.path ?? null });
  }

  protected async setVariableRequest(
    response: DebugProtocol.SetVariableResponse,
    args: DebugProtocol.SetVariableArguments,
  ): Promise<void> {
    let value: unknown = args.value;
    try {
      value = JSON.parse(args.value);
    } catch {
      /* a bare string */
    }
    try {
      const status = await this.bridge!.request<Status>("set", { name: args.name, value });
      this.apply(status, false);
      const c = (await this.state()).find((x) => x.name === args.name)!;
      response.body = { value: preview(c), type: c.dtype, variablesReference: this.handles.create({ kind: "column", name: args.name }) };
      this.sendResponse(response);
    } catch (e) {
      this.sendErrorResponse(response, 2, (e as Error).message);
    }
  }

  protected async evaluateRequest(
    response: DebugProtocol.EvaluateResponse,
    args: DebugProtocol.EvaluateArguments,
  ): Promise<void> {
    const c = (await this.state().catch(() => [])).find((x) => x.name === args.expression.trim());
    if (!c) return this.sendErrorResponse(response, 3, `not a column: ${args.expression}`);
    response.body = { result: preview(c), type: c.dtype, variablesReference: this.handles.create({ kind: "column", name: c.name }) };
    this.sendResponse(response);
  }

  protected async customRequest(command: string, response: DebugProtocol.Response, args: Record<string, unknown>): Promise<void> {
    try {
      switch (command) {
        case "decider.describe":
          response.body = this.describe;
          break;
        case "decider.state":
          response.body = { columns: this.current || this.finished ? await this.state() : null };
          break;
        case "decider.info":
          response.body = { debugpyPort: this.debugpyPort, current: this.current, node: this.current ? this.nodes.get(this.current.path) : null };
          break;
        case "decider.lineage":
          response.body = await this.lineage(args.name as string);
          break;
        default:
          return this.sendErrorResponse(response, 4, `unknown request ${command}`);
      }
      this.sendResponse(response);
    } catch (e) {
      this.sendErrorResponse(response, 5, (e as Error).message);
    }
  }
}

function preview(c: ColumnSummary): string {
  const shown = c.preview.map((v) => JSON.stringify(v)).join(", ");
  return `[${shown}${c.rows > c.preview.length ? ", …" : ""}]${c.nulls ? ` ${c.nulls} null` : ""}`;
}

function samePath(a: string, b: string): boolean {
  return path.resolve(a) === path.resolve(b);
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

function optionsFromEnv(): AdapterOptions {
  return { python: (process.env.DECIDER_PYTHON ?? "python3").split(" "), debugpyLibs: process.env.DECIDER_DEBUGPY_LIBS };
}
