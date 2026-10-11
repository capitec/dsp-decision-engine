"""Cold load in headless Chromium: serve <dir>, open its index.html in a new browser context per run (cold cache)
and read `window.__ready`, which the page sets to `performance.now()` once the approach is initialised (ms since
navigation start, so it includes fetching, compiling and instantiating the wasm). Prints the median of N runs.

    uv run --no-project --with playwright python coldload_browser.py <dir> <port> [-n 20]
"""

import argparse
import functools
import json
import statistics
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript",
                      ".js": "text/javascript"}

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("port", type=int)
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=args.dir))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    times = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for _ in range(args.n):
                context = browser.new_context()
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{args.port}/index.html")
                page.wait_for_function("window.__ready !== undefined", timeout=30_000)
                times.append(page.evaluate("window.__ready"))
                context.close()
            browser.close()
    finally:
        server.shutdown()
    print(json.dumps({"dir": args.dir, "n": len(times), "median_ms": round(statistics.median(times), 3),
                      "min_ms": round(min(times), 3), "max_ms": round(max(times), 3)}))


if __name__ == "__main__":
    main()
