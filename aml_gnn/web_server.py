"""
web_server.py — open the static web app locally.

  python -m aml_gnn web [--name N] [--port 8080]

Serves web/ (or runs/<name>/web/ for a scored dataset) on 127.0.0.1 only,
and opens it in the default browser.
"""

import functools
import http.server
import threading
import webbrowser
from pathlib import Path

# Explicit types: on some Windows setups the registry maps .js to text/plain,
# and browsers refuse to run ES modules served that way.
MIME_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".md": "text/markdown",
    ".svg": "image/svg+xml",
}


class AppHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **MIME_TYPES}

    def log_message(self, format, *args):      # keep the terminal readable
        pass


def make_server(root: Path, port: int = 8080) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(AppHandler, directory=str(root))
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(root: Path, port: int = 8080, open_browser: bool = True) -> None:
    if not (root / "data" / "summary.json").exists():
        raise SystemExit(f"No app data in {root / 'data'} — run `python -m aml_gnn export-web` first.")
    try:
        server = make_server(root, port)
    except OSError as error:
        raise SystemExit(f"Port {port} is busy ({error}); try --port 8090") from error

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Serving {root}\nOpen {url}  —  press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped.")
