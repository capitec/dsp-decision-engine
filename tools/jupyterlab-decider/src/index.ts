import type { JupyterFrontEnd, JupyterFrontEndPlugin } from "@jupyterlab/application";
import { Notification } from "@jupyterlab/apputils";
import { IDocumentManager } from "@jupyterlab/docmanager";
import { IDefaultFileBrowser } from "@jupyterlab/filebrowser";
import { INotebookTracker } from "@jupyterlab/notebook";
import type { Kernel } from "@jupyterlab/services";
import { Host, type Target } from "./host";
import { flowPanel } from "./panel";

const COMMAND = "decider:open-flow";
/** The comm target both sides register: the kernel for flow files, the page for `debug(...)` in a notebook. */
const TARGET = "decider";

const dirname = (p: string) => p.slice(0, Math.max(p.lastIndexOf("/"), 0));
const basename = (p: string) => p.slice(p.lastIndexOf("/") + 1);

const plugin: JupyterFrontEndPlugin<void> = {
  id: "jupyterlab-decider:plugin",
  description: "Visualise, run and step through decider pipelines",
  autoStart: true,
  requires: [IDocumentManager, IDefaultFileBrowser, INotebookTracker],
  activate: (app: JupyterFrontEnd, docs: IDocumentManager, browser: IDefaultFileBrowser, notebooks: INotebookTracker) => {
    const notify = (message: string) => void Notification.error(`decider: ${message}`, { autoClose: false });

    /** A panel on `target`'s flow, over `comm`; `dir` is the kernel's working directory under the server root. */
    const show = async (comm: Kernel.IComm, target: Target, dir: string, onClose?: () => void) => {
      let root = "";
      const reveal = async (file: string, line: number | null) => {
        if (!file.startsWith(`${root}/`)) return notify(`${file} is outside the folder Jupyter serves`);
        const w = docs.openOrReveal(file.slice(root.length + 1), undefined, undefined, { mode: "split-left", ref: panel.id }) as any;
        await w?.context.ready;
        const at = { line: Math.max((line ?? 1) - 1, 0), column: 0 };
        w?.content.editor?.setCursorPosition(at);
        w?.content.editor?.revealPosition(at);
      };
      const host = new Host(comm, target, { notify, reveal: (file, line) => void reveal(file, line) });
      const panel = flowPanel(host, target.pipeline ?? basename(target.file!), notify);
      panel.disposed.connect(() => {
        if (!comm.isDisposed) comm.close();
        onClose?.();
      });
      app.shell.add(panel, "main", { mode: "split-right" });
      const cwd = await host.request<string>("cwd");
      root = dir ? cwd.slice(0, -(dir.length + 1)) : cwd;
      await host.open();
    };

    // A flow file runs in a kernel of its own, started in the file's folder, which goes when the panel closes.
    const openFile = async (path: string) => {
      const session = await app.serviceManager.sessions.startNew({
        path,
        type: "decider",
        name: basename(path),
        kernel: { name: app.serviceManager.kernelspecs.specs?.default ?? "python3" },
      });
      const kernel = session.kernel!;
      const reply = await kernel.requestExecute({ code: "import decider_jupyter", silent: true, store_history: false }).done;
      if (reply.content.status !== "ok") {
        void session.shutdown();
        return notify(`the kernel can't import decider_jupyter; install jupyterlab-decider in its environment (${(reply.content as any).evalue ?? ""})`);
      }
      const comm = kernel.createComm(TARGET);
      comm.open();
      await show(comm, { file: basename(path) }, dirname(path), () => void session.shutdown());
    };

    app.commands.addCommand(COMMAND, {
      label: "Debug decider flow",
      caption: "Show the pipeline in this file, and run and step through it",
      execute: async (args) => {
        const path = (args.path as string | undefined) ?? browser.selectedItems().next()?.value?.path;
        if (path) await openFile(path).catch((e) => notify((e as Error).message));
      },
    });
    app.contextMenu.addItem({ command: COMMAND, selector: '.jp-DirListing-item[data-file-type="python"]', rank: 3 });

    notebooks.widgetAdded.connect((_, nb) => {
      const register = () =>
        nb.sessionContext.session?.kernel?.registerCommTarget(TARGET, (comm, msg) => {
          const { pipeline } = msg.content.data as { pipeline: string };
          show(comm, { file: null, pipeline }, dirname(nb.context.path)).catch((e) => notify((e as Error).message));
        });
      nb.sessionContext.kernelChanged.connect(register);
      register();
    });
  },
};

export default plugin;
