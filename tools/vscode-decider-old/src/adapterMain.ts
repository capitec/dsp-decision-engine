// Standalone stdio entry, so the adapter can be driven by DebugClient in tests
// (and by other editors). VS Code itself uses the inline implementation.
import { DebugSession } from "@vscode/debugadapter";
import { DeciderDebugSession } from "./adapter";

DebugSession.run(DeciderDebugSession as unknown as typeof DebugSession);
