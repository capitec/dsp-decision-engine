import { execFile, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

async function git(cwd: string, ...args: string[]): Promise<string> {
  return (await run("git", args, { cwd, maxBuffer: 16 << 20 })).stdout.trim();
}

export function repoRoot(file: string): Promise<string> {
  return git(path.dirname(file), "rev-parse", "--show-toplevel");
}

export interface Ref {
  ref: string;
  label: string;
  description: string;
}

/** Tags, local branches and recent commits, newest first, for picking a baseline. */
export async function listRefs(root: string, commits = 20): Promise<Ref[]> {
  const refs = await git(root, "for-each-ref", "--sort=-creatordate", "--format=%(refname:short)\t%(objectname:short)\t%(refname)", "refs/tags", "refs/heads");
  const log = await git(root, "log", `-${commits}`, "--format=%h\t%s\t%cr");
  const named = refs
    .split("\n")
    .filter(Boolean)
    .map((l) => {
      const [name, sha, full] = l.split("\t");
      return { ref: name, label: name, sha, description: full.startsWith("refs/tags/") ? "tag" : "branch" };
    });
  const recent = log
    .split("\n")
    .filter(Boolean)
    .map((l) => {
      const [sha, subject, when] = l.split("\t");
      return { ref: sha, label: sha, description: `${subject} (${when})` };
    });
  const head = recent[0] ? [{ ref: "HEAD", label: "HEAD", sha: recent[0].ref, description: `last commit: ${recent[0].description}` }] : [];
  // Several names often point at one commit; say so, so the list doesn't look like different choices.
  const names = new Map<string, string[]>();
  for (const r of [...head, ...named]) names.set(r.sha, [...(names.get(r.sha) ?? []), r.label]);
  const alias = (r: { sha: string; label: string }) => (names.get(r.sha) ?? []).filter((n) => n !== r.label);
  return [
    ...[...head, ...named].map(({ sha, ...r }) => ({ ...r, description: alias({ sha, label: r.label }).length ? `${r.description} · same commit as ${alias({ sha, label: r.label }).join(", ")}` : r.description })),
    ...recent.map((r) => ({ ...r, description: names.has(r.ref) ? `${r.description} · ${names.get(r.ref)!.join(", ")}` : r.description })),
  ];
}

/**
 * The tree of `ref` as plain files, extracted once per commit into a cache
 * directory. Returns that directory; nothing in the working tree changes.
 */
export async function materialise(root: string, ref: string, cache = path.join(os.tmpdir(), "decider-revisions")): Promise<string> {
  const sha = await git(root, "rev-parse", "--verify", `${ref}^{commit}`);
  const dir = path.join(cache, sha);
  if (fs.existsSync(dir)) return dir;
  const partial = `${dir}.partial-${process.pid}`;
  fs.mkdirSync(partial, { recursive: true });
  await new Promise<void>((resolve, reject) => {
    const archive = spawn("git", ["archive", "--format=tar", sha], { cwd: root });
    const tar = spawn("tar", ["-x", "-C", partial]);
    archive.stdout.pipe(tar.stdin);
    let err = "";
    archive.stderr.on("data", (d) => (err += d));
    tar.stderr.on("data", (d) => (err += d));
    tar.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`git archive ${ref}: ${err || `tar exited ${code}`}`))));
    archive.on("error", reject);
    tar.on("error", reject);
  });
  fs.renameSync(partial, dir);
  return dir;
}
