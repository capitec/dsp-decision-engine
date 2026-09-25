import * as path from "node:path";
import { Breakpoint, Event, Handles, InitializedEvent, LoggingDebugSession, OutputEvent, Scope, Source, StackFrame, StoppedEvent, TerminatedEvent, Thread, Variable } from "@vscode/debugadapter";
import type { DebugProtocol } from "@vscode/debugprotocol";
import { Bridge, freePort } from "./bridge";
import { kindLabel, lastSegment, formatValue, previewOf, readEvents, setFields, walk } from "@decider/ui";
import type { Checkpoint, ColumnSummary, Controls, DescribeResult, Hit, IRNodeJson, Lineage, RecordKey, RunStatus, Status, Visits } from "@decider/ui";
import { nodeAtLine } from "./sourceMap";
import { optionsFromEnv, type AdapterOptions, type LaunchArgs } from "./launchArgs";

type VarRef =
  | { kind: "scope"; names: string[] | "all" }
  | { kind: "column"; name: string }
  | { kind: "values"; name: string }
  | { kind: "versions"; name: string }
  | { kind: "lineage"; entry: Lineage }
  | { kind: "visits"; path: string };

const THREAD = 1;

/**
 * Maps the Debug Adapter Protocol onto a decider session: one thread, a
 * stack made of the current node's ancestors, scopes for the node's inputs,
 * outputs and the whole state, `setVariable` as `set`, and `restartFrame` as
 * `rewind`.
 */
export class DeciderDebugSession extends LoggingDebugSession {
  private opts: AdapterOptions;
  private bridge?: Bridge;
  private describe?: DescribeResult;
  private nodes = new Map<string, IRNodeJson>();
  private parents = new Map<string, string>();
  private current: Checkpoint | null = null;
  private finished = false;
  private finishedPaths: string[] = [];
  private visits: Visits = {};
  private edits: Record<string, "delete" | "replace"> = {};
  private record: number | null = null;
  private hit: Hit | null = null;
  private handles = new Handles<VarRef>();
  private frameIds = new Map<number, string>();
  private stateCache?: Promise<{ columns: ColumnSummary[]; key: RecordKey }>;
  private sourceBps = new Map<string, string[]>();
  private fnBps: string[] = [];
  private applied = new Set<string>();
  private configured!: () => void;
  private configuredPromise = new Promise<void>((r) => (this.configured = r));
  private launchArgs?: LaunchArgs;
  private debugpyPort?: number;

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
    this.record = args.record ?? null;
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
      setFields(this.describe.fields);
      walk(this.describe.ir, (n, parent) => {
        this.nodes.set(n.path, n);
        if (parent) this.parents.set(n.path, parent.path);
      });
      this.sendResponse(response);
      // Breakpoints arrive after this event; launch continues once configurationDone lands.
      this.sendEvent(new InitializedEvent());
      await this.configuredPromise;
      await this.startSession();
      this.sendEvent(new OutputEvent(`decider: running ${this.describe.pipeline} from ${args.program}\n`, "console"));
      await this.run(args.stopOnEntry === false ? "resume" : "step_into", {}, args.stopOnEntry === false ? undefined : "entry");
    } catch (e) {
      this.sendErrorResponse(response, 1, `decider: ${(e as Error).message}`);
      this.sendEvent(new TerminatedEvent());
    }
  }

  private async startSession() {
    const a = this.launchArgs!;
    this.finishedPaths = [];
    this.visits = {};
    this.edits = {};
    const status = await this.bridge!.request<Status>("start", {
      file: a.program,
      pipeline: a.pipeline,
      data: a.data ?? null,
      params: a.params ?? null,
      breakpoints: this.desiredBreakpoints(),
      forces: a.controls?.forces ?? [],
      watches: a.controls?.watches ?? [],
    });
    this.applied = new Set(this.desiredBreakpoints());
    this.apply(status, false);
  }

  private desiredBreakpoints(): string[] {
    return [...new Set([...this.fnBps, ...[...this.sourceBps.values()].flat()])];
  }

  private async syncBreakpoints() {
    if (!this.bridge) return;
    const desired = new Set(this.desiredBreakpoints());
    for (const p of desired) if (!this.applied.has(p)) await this.bridge.request("break_at", { target: p });
    for (const p of this.applied) if (!desired.has(p)) await this.bridge.request("clear_break", { target: p });
    this.applied = desired;
  }

  protected async setBreakPointsRequest(
    response: DebugProtocol.SetBreakpointsResponse,
    args: DebugProtocol.SetBreakpointsArguments,
  ): Promise<void> {
    const file = args.source.path ?? "";
    const paths: string[] = [];
    response.body = {
      breakpoints: (args.breakpoints ?? []).map((bp) => {
        const anchor = nodeAtLine(this.nodes.values(), file, bp.line);
        if (!anchor) return new Breakpoint(false);
        paths.push(anchor.path);
        const b: DebugProtocol.Breakpoint = new Breakpoint(true, anchor.line, undefined, new Source(path.basename(file), file));
        b.message = anchor.path;
        return b as Breakpoint;
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

  private async run(cmd: string, args: Record<string, unknown> = {}, reason?: string): Promise<void> {
    let status: Status;
    try {
      status = await this.bridge!.request<Status>(cmd, args);
    } catch (e) {
      this.sendEvent(new OutputEvent(`decider: ${(e as Error).message}\n`, "stderr"));
      this.sendEvent(new StoppedEvent("exception", THREAD, (e as Error).message));
      return;
    }
    this.apply(status, true, reason);
  }

  private apply(status: Status, stop: boolean, reason?: string) {
    const read = readEvents(status.events, this.nodes, this.visits, this.finishedPaths, this.edits);
    for (const [text, category] of read.lines) this.sendEvent(new OutputEvent(text, category));
    const paused = reason ?? read.paused;
    // A rewind un-runs steps; the bridge knows which have run since.
    if (status.ran) this.finishedPaths = status.ran;
    this.current = status.current;
    this.finished = status.finished;
    this.hit = status.hit ?? null;
    if (this.hit) this.sendEvent(new OutputEvent(`decider: paused on ${this.hit.text}\n`, "console"));
    this.refresh();
    if (!stop) return;
    if (status.error) this.sendEvent(new StoppedEvent("exception", THREAD, status.error));
    else if (this.hit) this.sendEvent(new StoppedEvent("data breakpoint", THREAD, this.hit.text));
    else if (this.finished) this.sendEvent(new TerminatedEvent());
    else this.sendEvent(new StoppedEvent(paused ?? "step", THREAD));
  }

  /** Drop cached values and tell the views where the run is. */
  private refresh() {
    this.stateCache = undefined;
    this.handles.reset();
    const body: RunStatus = { current: this.current, finished: this.finished, finishedPaths: this.finishedPaths, visits: this.visits, record: this.record, edits: this.edits, hit: this.hit };
    this.sendEvent(new Event("decider.status", body));
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
    this.sendResponse(response);
    await this.run("step_out");
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
    await this.startSession();
    await this.run("step_into", {}, "entry");
  }

  protected async disconnectRequest(response: DebugProtocol.DisconnectResponse): Promise<void> {
    this.finished = true;
    await this.bridge?.dispose();
    this.sendResponse(response);
  }

  protected async terminateRequest(response: DebugProtocol.TerminateResponse): Promise<void> {
    await this.disconnectRequest(response);
    this.sendEvent(new TerminatedEvent());
  }

  protected threadsRequest(response: DebugProtocol.ThreadsResponse): void {
    response.body = { threads: [new Thread(THREAD, this.record === null ? "pipeline" : `pipeline (record ${this.record})`)] };
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
      // A step built inline (a branch inside flow(...)) has no line of its own; show where its parent is.
      let at: IRNodeJson | undefined = node;
      for (let up = p; at && !at.file && this.parents.has(up); up = this.parents.get(up)!) at = this.nodes.get(this.parents.get(up)!);
      const src = at?.file ? new Source(path.basename(at.file), at.file) : undefined;
      const when = id === 1 ? `  ${this.current!.when}` : "";
      const after = id === 1 && this.current!.when === "after" && node?.kind === "call" && node.python?.file === at?.file ? node.python?.endLine : null;
      frames.push(new StackFrame(id, `${lastSegment(p)}  [${node ? kindLabel(node) : "?"}]${when}`, src, after ?? at?.line ?? 0, 1));
      p = this.parents.get(p);
    }
    response.body = { stackFrames: frames, totalFrames: frames.length };
    this.sendResponse(response);
  }

  protected scopesRequest(response: DebugProtocol.ScopesResponse, args: DebugProtocol.ScopesArguments): void {
    const p = this.frameIds.get(args.frameId) ?? "";
    const node = this.nodes.get(p);
    const scopes: Scope[] = [];
    if (node?.kind === "call") {
      if (node.inputs) scopes.push(new Scope("Inputs", this.handles.create({ kind: "scope", names: node.inputs }), false));
      if (node.outputs) scopes.push(new Scope("Outputs", this.handles.create({ kind: "scope", names: node.outputs }), false));
      if (this.visits[p] && Object.keys(this.visits[p]).length) scopes.push(new Scope("Visited", this.handles.create({ kind: "visits", path: p }), false));
    }
    scopes.push(new Scope("State", this.handles.create({ kind: "scope", names: "all" }), true));
    response.body = { scopes };
    this.sendResponse(response);
  }

  private state(): Promise<ColumnSummary[]> {
    return this.fullState().then((r) => r.columns);
  }

  private fullState() {
    this.stateCache ??= this.bridge!.request<{ columns: ColumnSummary[]; key: RecordKey }>("state", { row: this.record });
    return this.stateCache;
  }

  protected async variablesRequest(
    response: DebugProtocol.VariablesResponse,
    args: DebugProtocol.VariablesArguments,
  ): Promise<void> {
    let variables: Variable[];
    try {
      variables = await this.variablesFor(this.handles.get(args.variablesReference));
    } catch (e) {
      variables = [new Variable("error", (e as Error).message)];
    }
    response.body = { variables };
    this.sendResponse(response);
  }

  /** A value for the Variables view; `declined` marks a declined record's offer figures, which aren't an offer. */
  private shown(c: ColumnSummary, declined = false): string {
    if (this.record === null) return previewOf(c, c.name);
    const offer = declined && typeof c.value === "number" && (this.describe?.outcome ?? []).includes(c.name);
    return `${formatValue(c.value, c.name)}${offer ? " · no offer" : ""}`;
  }

  private async variablesFor(ref: VarRef): Promise<Variable[]> {
    switch (ref.kind) {
      case "scope": {
        const cols = await this.state();
        const byName = new Map(cols.map((c) => [c.name, c]));
        const declined = byName.get("decision")?.value === "decline";
        const names = ref.names === "all" ? cols.map((c) => c.name) : ref.names;
        return names.map((n) => {
          const c = byName.get(n);
          if (!c) return new Variable(n, "<not yet computed>");
          const v: DebugProtocol.Variable = new Variable(n, this.shown(c, declined), this.handles.create({ kind: "column", name: n }));
          v.type = c.dtype;
          v.evaluateName = n;
          return v as Variable;
        });
      }
      case "column": {
        const c = (await this.state()).find((x) => x.name === ref.name)!;
        return [
          new Variable("values", this.record === null ? `${c.rows} rows` : `record ${this.record}`, this.handles.create({ kind: "values", name: ref.name })),
          new Variable("producer", c.producer),
          new Variable("nulls", String(c.nulls)),
          new Variable("versions", String(c.versions), this.handles.create({ kind: "versions", name: ref.name })),
          new Variable("lineage", "what it was computed from, so far", this.handles.create({ kind: "lineage", entry: await this.lineage(ref.name) })),
        ];
      }
      case "values": {
        const col = await this.bridge!.request<{ values: unknown[] }>("column", { name: ref.name });
        return col.values.map((v, i) => new Variable(`[${i}]`, JSON.stringify(v)));
      }
      case "versions": {
        const col = await this.bridge!.request<{ versions: { producer: string; values: unknown[] }[] }>("column", { name: ref.name, row: this.record });
        return col.versions.map((v, i) => new Variable(`${i} ${v.producer}`, JSON.stringify(v.values.slice(0, 5))));
      }
      case "lineage":
        return ref.entry.inputs.map((e) => {
          const label = `${e.producer ?? "<input column>"}${e.via ? ` (${e.via})` : ""}`;
          const value = this.record === null ? label : `${JSON.stringify(e.value)}  ← ${label}`;
          return new Variable(e.name, value, e.inputs.length ? this.handles.create({ kind: "lineage", entry: e }) : 0);
        });
      case "visits":
        return Object.entries(this.visits[ref.path] ?? {}).map(([loc, rows]) => new Variable(`#${loc}`, `${rows} rows`));
    }
  }

  private lineage(name: string): Promise<Lineage> {
    return this.bridge!.request<Lineage>("lineage", { name, row: this.record });
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
      if (status.error) throw new Error(status.error);
      this.apply(status, false);
      const c = (await this.state()).find((x) => x.name === args.name)!;
      response.body = { value: this.shown(c), type: c.dtype, variablesReference: this.handles.create({ kind: "column", name: args.name }) };
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
    response.body = { result: this.shown(c), type: c.dtype, variablesReference: this.handles.create({ kind: "column", name: c.name }) };
    this.sendResponse(response);
  }

  protected async customRequest(command: string, response: DebugProtocol.Response, args: Record<string, unknown> = {}): Promise<void> {
    try {
      switch (command) {
        case "decider.describe":
          response.body = this.describe;
          break;
        case "decider.state":
          response.body = this.current || this.finished ? { ...(await this.fullState()), record: this.record } : { columns: null, key: null, record: this.record };
          break;
        case "decider.info":
          response.body = { debugpyPort: this.debugpyPort, current: this.current, node: this.current ? this.nodes.get(this.current.path) : null };
          break;
        case "decider.lineage":
          response.body = await this.lineage(args.name as string);
          break;
        case "decider.treePath":
          response.body = await this.bridge!.request("tree_path", { path: args.path, row: this.record ?? 0 });
          break;
        case "decider.debugCondition":
          response.body = this.record === null ? { condition: null } : await this.bridge!.request("debug_condition", { path: args.path, row: this.record });
          break;
        case "decider.skip":
        case "decider.reloadStep":
        case "decider.restore": {
          // A wiring error rejects the request and leaves the run as it was.
          const cmd = command === "decider.skip" ? "skip" : command === "decider.restore" ? "restore" : "reload_step";
          const status = await this.bridge!.request<Status & { formula: string | null }>(cmd, { path: args.path });
          response.body = { formula: status.formula };
          const before = { ...this.edits };
          this.sendResponse(response);
          this.apply(status, false);
          // Putting the original back is an edit to the session (of the step, or of its flow), but not one to show.
          if (cmd === "restore") {
            delete before[args.path as string];
            this.edits = before;
          }
          this.refresh();
          if (this.finished) this.sendEvent(new TerminatedEvent());
          else this.sendEvent(new StoppedEvent("edit", THREAD));
          return;
        }
        case "decider.changes":
          response.body = await this.bridge!.request("changes", { name: args.name, row: this.record });
          break;
        case "decider.rerun":
          this.sendResponse(response);
          return void (await this.run("rerun", { path: args.path, back: args.back ?? null, when: args.when ?? "before" }, "goto"));
        case "decider.goTo":
          this.sendResponse(response);
          return void (await this.run("go_to", typeof args.change === "string" ? { path: args.change, row: this.record } : { change: args.change }, "goto"));
        case "decider.rewind":
          this.sendResponse(response);
          return void (await this.run("rewind", { path: args.path }));
        case "decider.restartWith":
          this.launchArgs!.params = args.params;
          this.sendResponse(response);
          await this.startSession();
          return void (await this.run("step_into", {}, "entry"));
        case "decider.compareEdits":
          response.body = await this.bridge!.request("compare_edits", { path: args.path ?? null });
          break;
        case "decider.sweep":
          response.body = await this.bridge!.request("sweep", { scenarios: args.scenarios, from_here: true });
          break;
        case "decider.setControls":
          // Kept for a restart; a session not started yet picks them up when it starts.
          this.launchArgs!.controls = args as unknown as Controls;
          // Nothing runs, so no status: the views keep their selection.
          if (this.current || this.finished) await this.bridge!.request("set_controls", args);
          break;
        case "decider.setRecord":
          this.record = typeof args.row === "number" ? args.row : null;
          this.refresh();
          // Re-stopping makes VS Code fetch the Variables view again, now for the record.
          this.sendEvent(new StoppedEvent("record", THREAD));
          response.body = { record: this.record };
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
