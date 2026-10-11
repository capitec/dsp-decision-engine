"""Cold load of the built tree editor in headless Chromium, builds interleaved run by run (a fresh context each).

Per build: ms since navigation start until the wasm is instantiated (a hook on WebAssembly records it; the build
without merge-core has none), until #root first has content, and until the load event. Medians of N runs.

    uv run --no-project --with playwright python coldload_app.py name=<build dir or http://url> ... [-n 10] [--port 9121]
"""

import argparse
import functools
import json
import os
import statistics
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

API = "http://localhost:9100"
HOOK = """
for (const f of ['instantiateStreaming', 'instantiate']) {
  const orig = WebAssembly[f];
  WebAssembly[f] = (...a) => orig.apply(WebAssembly, a).then((r) => { window.__wasm ??= performance.now(); return r; });
}
new MutationObserver((_, o) => {
  if (document.getElementById('root')?.childElementCount) { window.__root = performance.now(); o.disconnect(); }
}).observe(document, { childList: true, subtree: true });
"""


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}

    def do_GET(self):
        # The page asks the API for its login config before it renders: forward /api and /auth to the running stack.
        if self.path.startswith(("/api", "/auth")):
            try:
                with urllib.request.urlopen(f"{API}{self.path}") as r:
                    body, status = r.read(), r.status
            except urllib.error.HTTPError as e:
                body, status = e.read(), e.code
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            # Client-side routes (/login) are served the app.
            if not os.path.exists(self.translate_path(self.path)):
                self.path = "/"
            super().do_GET()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("builds", nargs="+")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--port", type=int, default=9121)
    args = parser.parse_args()
    urls, port = {}, args.port
    for spec in args.builds:
        name, target = spec.split("=", 1)
        if target.startswith("http"):
            urls[name] = target
            continue
        server = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=target))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        urls[name] = f"http://127.0.0.1:{port}/"
        port += 1
    times = {name: {"wasm": [], "root": [], "load": []} for name in urls}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for _ in range(args.n):
            for name, url in urls.items():
                page = browser.new_context().new_page()
                page.add_init_script(HOOK)
                page.goto(url, wait_until="load")
                try:
                    page.wait_for_function("window.__root !== undefined", timeout=8_000)
                except Exception:
                    print("FAILED", name, page.url, page.evaluate("document.body.innerHTML.slice(0, 200)"))
                    raise
                got = page.evaluate("""() => ({wasm: window.__wasm ?? null, root: window.__root,
                    load: performance.timing.loadEventEnd - performance.timing.navigationStart})""")
                for k, v in got.items():
                    if v is not None:
                        times[name][k].append(v)
                page.context.close()
        browser.close()
    for name, t in times.items():
        print(json.dumps({"build": name, "n": args.n, **{f"{k}_median_ms": round(statistics.median(v), 1) for k, v in t.items() if v}}))


if __name__ == "__main__":
    main()
