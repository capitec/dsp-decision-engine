import { useEffect, useMemo, useState } from "react";
import { callNodes, setFields, type DescribeResult, type FromUI, type ToUI } from "./model/protocol";
import { groupPaths } from "./layout";
import { Graph } from "./Graph";
import { Inspector, type Selection } from "./Inspector";
import { Structure } from "./Structure";

// Flows up to this many steps start fully open; bigger ones start folded, a group at a time.
const OPEN_ALL = 80;

/** Messages only an editor can act on: opening source. Controls that send any other are hidden. */
export type EditorMessage = "reveal";

export interface AppProps {
  send: (m: FromUI) => void;
  listen: (on: (m: ToUI) => void) => () => void;
  can: ReadonlySet<EditorMessage>;
}

/**
 * The whole flow view. The host renders it into a sized element and wires messages both ways:
 *
 *     createRoot(el).render(<App send={post} listen={subscribe} can={new Set(["reveal"])} />);
 */
export function App(props: AppProps) {
  return <div className="decider"><View {...props} /></div>;
}

function View({ send, listen, can }: AppProps) {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [selected, setSelected] = useState<Selection>();
  const [opened, setOpened] = useState<Set<string>>(new Set());

  useEffect(() => {
    const stop = listen((m) => {
      switch (m.type) {
        case "describe":
          setFields(m.describe.fields);
          setDescribe(m.describe);
          setSelected(undefined);
          setOpened(callNodes(m.describe.ir).length <= OPEN_ALL ? groupPaths(m.describe.ir) : new Set());
          break;
        case "select":
          // From the editor's Structure tree: select the node and open the groups around it.
          setSelected({ kind: "node", path: m.path });
          setOpened((o) => {
            const next = new Set(o);
            for (const p of ancestors(m.path)) next.add(p);
            return next;
          });
          break;
      }
    });
    send({ type: "ready" });
    return stop;
  }, []);

  const nodes = useMemo(() => (describe ? callNodes(describe.ir) : []), [describe]);
  const reveal = can.has("reveal") ? (path: string) => send({ type: "reveal", path }) : undefined;

  if (!describe) return <div className="empty">Loading the flow…</div>;

  const selectedPath = selected?.kind === "node" ? selected.path : undefined;
  const open = (path: string) => opened.has(path);
  const selectNode = (path: string) => {
    setSelected({ kind: "node", path });
    setOpened((o) => {
      const next = new Set(o);
      for (const p of ancestors(path)) next.add(p);
      return next;
    });
  };
  const onSelect = (s: Selection | undefined) => {
    if (s?.kind === "node") selectNode(s.path);
    else setSelected(s);
  };
  const toggle = (path: string) =>
    setOpened((o) => {
      const next = new Set(o);
      if (next.has(path)) {
        next.delete(path);
        if (selectedPath?.startsWith(`${path}/`)) setSelected(undefined);
      } else next.add(path);
      return next;
    });

  return (
    <div className="app">
      <header>
        <strong className="pipeline">{describe.pipeline}</strong>
        <span className="size-badge" title="The assembled flow: every node, and the steps among them">
          {describe.size.calls} steps · {describe.size.nodes} nodes
        </span>
        <Discovery describe={describe} />
      </header>
      <div className="body">
        <Structure ir={describe.ir} selected={selectedPath} opened={opened} onSelect={selectNode} onToggle={toggle} />
        <Graph ir={describe.ir} open={open} selected={selected} onSelect={onSelect} onToggle={toggle} onReveal={reveal} nodes={nodes} />
        <Inspector ir={describe.ir} selected={selected} onSelect={onSelect} onReveal={reveal} nodes={nodes} />
      </div>
    </div>
  );
}

/** The entry point among the file's candidates, with the ones that were not pipelines and why. */
function Discovery({ describe }: { describe: DescribeResult }) {
  const invalid = describe.pipelines.filter((c) => c.status === "invalid");
  return (
    <details className="discovery">
      <summary>Discovery</summary>
      <div className="discovery-body">
        <div className="muted">
          Entry point <strong>{describe.pipeline}</strong> · {describe.size.calls} steps assembled from {describe.size.nodes} nodes.
        </div>
        {describe.pipelines.filter((c) => c.status !== "invalid" && c.name !== describe.pipeline).map((c) => (
          <div key={c.name}>
            <strong>{c.name}</strong> <span className="muted">{c.kind === "build" || c.kind === "factory" ? "factory" : c.kind}</span>
          </div>
        ))}
        {invalid.map((c) => (
          <div key={c.name} className="candidate-invalid">
            <s>{c.name}</s> <span className="muted">{c.reason}</span>
          </div>
        ))}
        {describe.pipelines.filter((c) => c.status !== "invalid").length <= 1 && invalid.length === 0 && (
          <div className="muted">No other candidates in this file.</div>
        )}
      </div>
    </details>
  );
}

/** The group paths above `path`, nearest first. */
function ancestors(path: string): string[] {
  const parts = path.split("/");
  const out: string[] = [];
  for (let i = 1; i < parts.length; i++) out.push(parts.slice(0, i).join("/"));
  return out.reverse();
}
