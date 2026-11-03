import threading
import urllib.request

from aml_gnn.web_server import make_server


def test_serves_the_app_with_module_safe_content_types(tmp_path):
    (tmp_path / "js").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>app</title>")
    (tmp_path / "js" / "main.js").write_text("export const ok = true;")
    server = make_server(tmp_path, port=0)                     # any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/js/main.js") as response:
            assert response.headers["Content-Type"].startswith("text/javascript")
        with urllib.request.urlopen(f"{base}/") as response:
            assert response.status == 200 and b"<title>app</title>" in response.read()
    finally:
        server.shutdown()
        server.server_close()
