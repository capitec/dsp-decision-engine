import type { Kernel } from "@jupyterlab/services";
import {
  compareTraces,
  editLabel,
  readEvents,
  summariseSweep,
  walk,
  type Checkpoint,
  type ColumnSummary,
  type Controls,
  type DescribeResult,
  type Force,
  type FromUI,
  type Hit,
  type IRNodeJson,
  type Lineage,
  type RecordKey,
  type RunStatus,
  type Scenario,
  type Status,
  type SweepResponse,
  type ToUI,
  type TraceResult,
  type ValueHistory,
  type Visits,
} from "@decider/ui";

/** The pipeline a panel shows: a flow file (its path as the kernel sees it), or a notebook's pipeline by name. */
export interface Target {
  file: string | null;
  pipeline?: string;
}

/** One side of a comparison: what to run the pipeline with. */
interface Side {
  label: string;
  params?: unknown;
  overrides?: Record<string, unknown>;
  row?: number | null;
  forces?: Force[];
}

export interface HostOptions {
  /** Shows an error to the user. */
  notify: (message: string) => void;
  /** Opens a source file at a line. */
  reveal: (file: string, line: number | null) => void;
}

/**
 * The panel's side of a debug session: sends bridge requests over a kernel comm, keeps where
 * the run is, and answers the UI's messages the way VS Code's extension and debug adapter do.
 */
export class Host {
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private listeners = new Set<(m: ToUI) => void>();
  // Held until the UI says it is ready.
  private queue: ToUI[] | null = [];
  private nodes = new Map<string, IRNodeJson>();
  /** A debug run is going: started, and not finished or stopped. */
  private live = false;
  private current: Checkpoint | null = null;
  private finishedPaths: string[] = [];
  private visits: Visits = {};
  private edits: Record<string, "delete" | "replace"> = {};
  private record: number | null = null;
  private hit: Hit | null = null;
  private controls: Controls = { forces: [], watches: [] };
  private params: unknown = null;
  // "Run to" targets: gone once the run pauses there or ends, like run to cursor.
  private runTo = new Set<string>();

  constructor(private comm: Kernel.IComm, private target: Target, private opts: HostOptions) {
    comm.onMsg = (msg) => {
      const r = msg.content.data as { id: number; ok: boolean; result?: unknown; error?: string };
      const p = this.pending.get(r.id);
      if (!p) return;
      this.pending.delete(r.id);
      r.ok ? p.resolve(r.result) : p.reject(new Error(r.error));
    };
    comm.onClose = () => {
      for (const p of this.pending.values()) p.reject(new Error("the kernel closed the debugger's connection"));
      this.pending.clear();
      this.live = false;
    };
  }

  /** Send a bridge command; `fresh` runs it on a bridge of its own, leaving the debug session as it is. */
  request<T = any>(cmd: string, args: Record<string, unknown> = {}, fresh = false): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.comm.send({ id, cmd, ...args, ...(fresh ? { fresh: true } : {}) });
    });
  }

  listen = (on: (m: ToUI) => void) => {
    this.listeners.add(on);
    return () => void this.listeners.delete(on);
  };

  send = (m: FromUI) => {
    this.onUI(m).catch((e) => this.opts.notify((e as Error).message));
  };

  private post(m: ToUI) {
    if (this.queue) this.queue.push(m);
    else for (const on of this.listeners) on(m);
  }

  /** Describe the flow and show it. */
  async open() {
    const describe = await this.request<DescribeResult>("describe", { ...this.target });
    this.target.pipeline = describe.pipeline;
    this.nodes.clear();
    walk(describe.ir, (n) => this.nodes.set(n.path, n));
    this.post({ type: "describe", describe });
  }

  /** Start a debug run from the beginning (again, if one is going): paused on entry, or running to a breakpoint. */
  async start(stopOnEntry = true) {
    await this.open();
    this.finishedPaths = [];
    this.visits = {};
    this.edits = {};
    this.record = null;
    await this.request("start", { ...this.target, data: null, params: this.params, breakpoints: [...this.runTo], ...this.controls });
    this.live = true;
    await this.run(stopOnEntry ? "step_into" : "resume");
  }

  /** Continue, step over, into or out: or start a run, when none is going. */
  async move(cmd: "resume" | "step" | "step_into" | "step_out") {
    if (this.live) await this.run(cmd);
    else if (cmd !== "step_out") await this.start(cmd !== "resume");
  }

  stop() {
    if (this.live) this.end();
  }

  private async run(cmd: string, args: Record<string, unknown> = {}) {
    await this.apply(await this.request<Status>(cmd, args));
  }

  private fold(status: Status) {
    readEvents(status.events, this.nodes, this.visits, this.finishedPaths, this.edits);
    // A rewind un-runs steps; the bridge knows which have run since.
    if (status.ran) this.finishedPaths = status.ran;
    this.current = status.current;
    this.hit = status.hit ?? null;
    if (status.error) this.opts.notify(status.error);
  }

  private async apply(status: Status) {
    this.fold(status);
    const at = this.current?.path;
    if (at !== undefined && this.runTo.delete(at)) await this.request("clear_break", { target: at });
    await (status.finished ? this.end() : this.show());
  }

  /** Tell the UI where the run is, and the values there. */
  private async show() {
    const run: RunStatus = { current: this.current, finished: false, finishedPaths: this.finishedPaths, visits: this.visits, record: this.record, edits: this.edits, hit: this.hit };
    this.post({ type: "status", ...run });
    const { columns, key } = await this.request<{ columns: ColumnSummary[] | null; key: RecordKey }>("state", { row: this.record });
    this.post({ type: "state", columns, rows: columns?.[0]?.rows ?? 0, key });
  }

  /** A run that finished or was stopped: like VS Code ending its debug session. */
  private end() {
    this.live = false;
    this.current = null;
    this.record = null;
    this.runTo.clear();
    this.post({ type: "status", current: null, finished: true, finishedPaths: [], visits: {}, record: null });
    this.post({ type: "state", columns: null, rows: 0, key: null });
  }

  private async onUI(m: FromUI) {
    switch (m.type) {
      case "ready":
        for (const q of this.queue?.splice(0) ?? []) for (const on of this.listeners) on(q);
        this.queue = null;
        break;
      case "reveal": {
        const node = this.nodes.get(m.path);
        if (node?.file) this.opts.reveal(node.file, node.line);
        break;
      }
      case "record":
        this.record = m.row;
        if (this.live) await this.show();
        break;
      case "lineage": {
        if (!this.live) return this.post({ type: "lineage", lineage: null, history: null });
        // A value not written yet has no lineage, but its history still lists what will set it.
        const record = this.record;
        const [lineage, history] = await Promise.all([
          this.request<Lineage>("lineage", { name: m.name, row: record }).catch(() => null),
          this.request<ValueHistory>("changes", { name: m.name, row: record }),
        ]);
        // The run may have stopped, or moved to another record, meanwhile: the answer is about a record no longer shown.
        if (this.live && record === this.record) this.post({ type: "lineage", lineage, history });
        break;
      }
      case "treePath":
        if (this.live) this.post({ type: "treePath", ...(await this.request<{ path: string; row: number; visited: string[]; result?: unknown[] }>("tree_path", { path: m.path, row: this.record ?? 0 })) });
        break;
      case "rerun":
        if (this.live) await this.run("rerun", { path: m.path, back: m.back ?? null, when: m.when ?? "before" });
        break;
      case "goTo":
        if (this.live) await this.run("go_to", typeof m.change === "string" ? { path: m.change, row: this.record } : { change: m.change });
        break;
      case "rewind":
        if (this.live) await this.run("rewind", { path: m.path });
        break;
      case "compareEdits":
        if (this.live) await this.compareEdits(m.label, m.edits, m.path);
        break;
      case "skip":
      case "reloadStep":
      case "restore":
        if (this.live) await this.edit(m.type, m.path);
        break;
      case "restartWith":
        this.params = m.params;
        await this.start();
        break;
      case "whatIf":
        await this.compare({ label: "current params" }, { label: m.label, params: m.params, overrides: m.overrides, row: m.row, forces: m.forces });
        break;
      case "setControls":
        this.controls = m.controls;
        if (this.live) await this.request("set_controls", { ...this.controls });
        break;
      case "compareForces":
        await this.compare(m.a, m.b, true);
        break;
      case "sweep":
        await this.sweep(m.scenarios, m.fromHere && this.live);
        break;
      case "runTo":
        this.runTo.add(m.path);
        if (!this.live) return this.start(false);
        await this.request("break_at", { target: m.path });
        await this.run("resume");
        break;
      case "step":
        await this.move("step");
        break;
    }
  }

  private async edit(type: "skip" | "reloadStep" | "restore", path: string) {
    const cmd = type === "skip" ? "skip" : type === "restore" ? "restore" : "reload_step";
    let status: Status & { formula?: string | null };
    try {
      status = await this.request(cmd, { path });
    } catch (e) {
      // A wiring error leaves the run as it was.
      const verb = type === "skip" ? "skip" : type === "restore" ? "restore" : "swap in";
      return this.opts.notify(`Couldn't ${verb} ${path.split("/").pop()}: ${(e as Error).message}`);
    }
    const before = { ...this.edits };
    this.fold(status);
    // Putting the original back is an edit to the session, but not one to show.
    if (type === "restore") {
      delete before[path];
      this.edits = before;
    }
    await (status.finished ? this.end() : this.show());
    if (type !== "skip") this.post({ type: "edited", path, formula: status.formula ?? null, restored: type === "restore" });
  }

  private async compareEdits(label: string, edits: Record<string, "delete" | "replace">, path?: string) {
    this.post({ type: "tab", tab: "compare" });
    this.post({ type: "compare", comparison: null, busy: "Running the flow as started and as edited…" });
    try {
      const r = await this.request<{ a: TraceResult; b: TraceResult }>("compare_edits", { path: path ?? null });
      const comparison = compareTraces(r.a, r.b, "the flow as started", label);
      const others = Object.entries(edits).filter(([p]) => p !== path).map(editLabel);
      const scope = path ? (others.length ? `Only ${label} is applied; ${others.join(", ")} ${others.length === 1 ? "is" : "are"} not.` : "") : others.length > 1 ? `All ${others.length} edits are applied.` : "";
      comparison.note = `${scope} Both versions ran from the start to the end; your debug run is still paused where it was.`.trim();
      // Both runs share one description, so a swapped step's new code shows only through the edits made.
      for (const st of comparison.steps) {
        if (path && path !== st.path) continue;
        if (edits[st.path] === "replace") st.structural.push("code");
        if (edits[st.path] === "delete" && st.status === "not taken") st.status = "removed";
      }
      this.post({ type: "compare", comparison });
    } catch (e) {
      this.post({ type: "compare", comparison: null, error: (e as Error).message });
    }
  }

  /** Run both sides to the end and line them up; `a` reuses `b`'s input rows unless it overrides some. */
  private async compare(a: Side, b: Side, forced = false) {
    this.post({ type: "tab", tab: "compare" });
    this.post({ type: "compare", comparison: null, busy: `Running ${a.label} and ${b.label}…` });
    const args = (s: Side, data: unknown) => ({ ...this.target, data, params: s.params ?? null, overrides: s.overrides ?? null, row: s.row ?? null, forces: s.forces ?? [] });
    try {
      const tb = await this.request<TraceResult>("trace", args(b, null), true);
      const ta = await this.request<TraceResult>("trace", args(a, a.overrides ? null : tb.data), true);
      const comparison = compareTraces(ta, tb, a.label, b.label);
      if (a.params || b.params) comparison.paramsDocs = { a: a.params ?? {}, b: b.params ?? {} };
      comparison.forced = forced || !!(a.forces?.length || b.forces?.length);
      const rows = [...(a.forces ?? []), ...(b.forces ?? [])].map((f) => f.row).filter((r): r is number => r !== null && r !== undefined);
      if (rows.length) comparison.forcedRows = [...new Set(rows)];
      this.post({ type: "compare", comparison });
    } catch (e) {
      this.post({ type: "compare", comparison: null, error: (e as Error).message });
    }
  }

  /** Run scenarios next to the unchanged run: forked from the paused session, or from the start. */
  private async sweep(list: Scenario[], fromHere: boolean) {
    this.post({ type: "tab", tab: "scenarios" });
    this.post({ type: "sweep", sweep: null, busy: `Running ${list.length} scenario${list.length === 1 ? "" : "s"}…` });
    try {
      const r = fromHere
        ? await this.request<SweepResponse>("sweep", { scenarios: list, from_here: true })
        : await this.request<SweepResponse>("sweep", { scenarios: list, from_here: false, ...this.target }, true);
      this.post({ type: "sweep", sweep: summariseSweep(r, list) });
    } catch (e) {
      this.post({ type: "sweep", sweep: null, error: (e as Error).message });
    }
  }
}
