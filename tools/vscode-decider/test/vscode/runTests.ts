// Launches VS Code / VSCodium with the extension and runs test/vscode/index.ts inside it.
import * as path from "node:path";
import { runTests } from "@vscode/test-electron";

async function main() {
  const extensionDevelopmentPath = path.resolve(__dirname, "..");
  const extensionTestsPath = path.resolve(__dirname, "vscode", "index.js");
  const workspace = path.join(extensionDevelopmentPath, "examples");
  await runTests({
    // The Electron binary itself: the /usr/bin/codium wrapper detaches and returns at once.
    vscodeExecutablePath: process.env.VSCODE_EXE ?? "/usr/share/codium/codium",
    extensionDevelopmentPath,
    extensionTestsPath,
    launchArgs: [workspace, "--disable-extensions", "--disable-workspace-trust", "--disable-gpu"],
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
