"""A one-page browser front end for the CLI.

    python scripts/serve.py

Runs the same commands documented in SOLUTION.md as subprocesses and shows
their output. It is deliberately not a second implementation of the pipeline:
every button shells out to `main.py` or `scripts/demo.py`, so what you read on
the page is exactly what the command line produces, and there is no second code
path to keep in sync.

Standard library only -- `http.server` and `subprocess`. A web framework would
be a dependency, a config file, and a deployment story for something that is one
handler and three commands.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_flow.config import INVOICE_DIR  # noqa: E402
from invoice_flow.tools.loaders import discover_invoices  # noqa: E402

REPORT_PATH = ROOT / "out" / "report.html"

# A run can involve a dozen LLM calls against a live API.
RUN_TIMEOUT_SECONDS = 600


def allowed_invoices() -> dict[str, Path]:
    """Invoice name -> path, for every file the loader accepts.

    The browser only ever sends a *name*, which is looked up here. Nothing the
    client sends reaches the command line as a path, so a crafted request cannot
    walk out of the invoice directory or name an arbitrary file.
    """
    return {path.name: path for path in discover_invoices(INVOICE_DIR)}


def build_command(action: str, invoice: str | None) -> list[str]:
    """Map an action to the exact documented command, or raise."""
    invoices = allowed_invoices()

    if action == "demo_all":
        return [sys.executable, "scripts/demo.py"]

    if action == "reset_db":
        return [sys.executable, "scripts/init_db.py", "--reset"]

    if action not in {"process", "demo_one"}:
        raise ValueError(f"Unknown action: {action!r}")

    if invoice not in invoices:
        raise ValueError(f"Not a known invoice: {invoice!r}")
    path = invoices[invoice].relative_to(ROOT).as_posix()

    if action == "process":
        return [sys.executable, "main.py", f"--invoice_path={path}"]
    return [
        sys.executable,
        "scripts/demo.py",
        "--invoice",
        path,
        "--report",
        "out/report.html",
    ]


def run_command(command: list[str]) -> dict[str, object]:
    """Run one command from the project root and capture everything it says."""
    env = {
        **os.environ,
        # Rich writes box-drawing characters; without this the child can fall
        # back to the console codepage on Windows and mangle its own output.
        "PYTHONIOENCODING": "utf-8",
    }
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RUN_TIMEOUT_SECONDS,
            env=env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        output = f"Timed out after {RUN_TIMEOUT_SECONDS}s."
        exit_code = None

    return {
        "command": " ".join(command[1:]),  # the interpreter path is noise
        "output": output.rstrip() or "(no output)",
        "exit_code": exit_code,
        "seconds": round(time.perf_counter() - started, 1),
    }


# Raw: the body is JavaScript source, so escape sequences like \n must survive
# to the browser rather than being interpreted by Python. Without this a `\n`
# written inside a JS string literal becomes a real line break, the literal
# never closes, and the whole <script> fails to parse -- which disables every
# button on the page, not just the one being edited.
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invoice pipeline</title>
<style>
  :root {
    --bg: #f6f7f9; --surface: #fff; --border: #dfe3e8; --text: #16181d;
    --muted: #616a76; --accent: #1f4f8f; --ok: #17794a; --warn: #8a5a00; --bad: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --surface: #1c1f25; --border: #2d323b; --text: #e8eaed;
      --muted: #9aa2b1; --accent: #7aa2f7; --ok: #6cc496; --warn: #e0b25c; --bad: #f28b82;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--text);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 62rem; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 .3rem; }
  .sub { color: var(--muted); margin: 0 0 2rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.1rem 1.25rem; margin-bottom: 1rem;
  }
  .card h2 { font-size: .95rem; margin: 0 0 .35rem; }
  .card p { margin: 0 0 .9rem; color: var(--muted); font-size: .89rem; }
  code {
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .85em;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: .1rem .3rem;
  }
  .controls { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; }
  button, select {
    font: inherit; font-size: .9rem; padding: .5rem .9rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--surface); color: var(--text); cursor: pointer;
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.danger { color: var(--bad); border-color: var(--bad); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #status { margin: 1.25rem 0 .5rem; font-size: .9rem; color: var(--muted); min-height: 1.6em; }
  #status .ok { color: var(--ok); font-weight: 600; }
  #status .warn { color: var(--warn); font-weight: 600; }
  #status .bad { color: var(--bad); font-weight: 600; }
  pre {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; overflow-x: auto; white-space: pre; margin: 0;
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; line-height: 1.5;
  }
  #reportLink { display: none; margin-top: .75rem; font-size: .9rem; }
</style>
</head>
<body>
<div class="wrap">

  <h1>Invoice processing pipeline</h1>
  <p class="sub">Each button runs the documented command and shows its output. Nothing is reimplemented here.</p>

  <div class="card">
    <h2>1 &middot; Process one invoice for real</h2>
    <p>Runs <code>python main.py --invoice_path=&hellip;</code> through the live Grok agents. Needs an API key in <code>.env</code>; takes about 20 seconds.</p>
    <div class="controls">
      <select id="invoice">__OPTIONS__</select>
      <button class="primary" data-action="process">Process invoice</button>
    </div>
  </div>

  <div class="card">
    <h2>2 &middot; Demo &mdash; every sample invoice</h2>
    <p>Runs <code>python scripts/demo.py</code> over all 20 files with scripted agent responses. No API key needed.</p>
    <div class="controls">
      <button data-action="demo_all">Run the full sweep</button>
    </div>
  </div>

  <div class="card">
    <h2>3 &middot; Demo &mdash; one invoice, with an HTML report</h2>
    <p>Runs <code>python scripts/demo.py --invoice &hellip; --report out/report.html</code> on the invoice selected above. No API key needed.</p>
    <div class="controls">
      <button data-action="demo_one">Run and build the report</button>
    </div>
    <div id="reportLink"><a href="/report" target="_blank" rel="noopener">Open the HTML report &rarr;</a></div>
  </div>

  <div class="card">
    <h2>4 &middot; Reset the inventory database</h2>
    <p>Runs <code>python scripts/init_db.py --reset</code>: deletes <code>data/inventory.db</code> and rebuilds it with the seed stock. <strong>This clears the payment ledger</strong>, so previously paid invoices become payable again and duplicate detection starts from nothing. Useful when re-running the same invoice for a demo.</p>
    <div class="controls">
      <button class="danger" data-action="reset_db">Reset database</button>
    </div>
  </div>

  <div id="status">Ready.</div>
  <pre id="output">Pick an action above.</pre>

</div>

<script>
  const statusEl = document.getElementById('status');
  const outputEl = document.getElementById('output');
  const reportLink = document.getElementById('reportLink');
  const buttons = document.querySelectorAll('button[data-action]');

  // Exit 1 from main.py means REJECTED or ESCALATED -- a correct outcome, not a
  // fault. Painting it the same red as a crash would teach the reader to
  // distrust the one signal the pipeline exists to produce.
  function verdictFor(action, code) {
    if (code === 0) {
      if (action === 'process') return '<span class="ok">APPROVED</span>';
      if (action === 'reset_db') return '<span class="ok">Database reset</span>';
      return '<span class="ok">Done</span>';
    }
    if (action === 'process' && code === 1) {
      return '<span class="warn">Not approved &mdash; rejected or escalated</span>';
    }
    return '<span class="bad">Exit ' + code + '</span>';
  }

  async function run(action) {
    const invoice = document.getElementById('invoice').value;

    if (action === 'reset_db' && !confirm(
      'Delete data/inventory.db and rebuild it from the seed stock?\n\n'
      + 'This clears the payment ledger: invoices already paid become payable '
      + 'again, and duplicate detection starts from nothing.'
    )) return;

    buttons.forEach(b => b.disabled = true);
    reportLink.style.display = 'none';
    statusEl.textContent = 'Running… this can take a while on a live run.';
    outputEl.textContent = '';

    try {
      const response = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, invoice })
      });
      const data = await response.json();

      if (data.error) {
        statusEl.innerHTML = '<span class="bad">Failed.</span> ' + data.error;
        outputEl.textContent = '';
        return;
      }

      outputEl.textContent = data.output;
      statusEl.innerHTML = verdictFor(action, data.exit_code)
        + ' &middot; ' + data.seconds + 's &middot; <code>' + data.command + '</code>';

      if (action === 'demo_one') reportLink.style.display = 'block';
    } catch (err) {
      statusEl.innerHTML = '<span class="bad">Could not reach the server.</span> ' + err;
    } finally {
      buttons.forEach(b => b.disabled = false);
    }
  }

  buttons.forEach(b => b.addEventListener('click', () => run(b.dataset.action)));
</script>
</body>
</html>
"""


def render_page() -> str:
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in allowed_invoices()
    )
    return PAGE.replace("__OPTIONS__", options)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        """One tidy line per request instead of the default two."""
        print(f"  {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's required spelling
        if self.path in ("/", "/index.html"):
            self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/report":
            if REPORT_PATH.is_file():
                self._send(
                    200, REPORT_PATH.read_bytes(), "text/html; charset=utf-8"
                )
            else:
                self._send(
                    404,
                    b"No report yet. Run action 3 first.",
                    "text/plain; charset=utf-8",
                )
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - http.server's required spelling
        if self.path != "/run":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            command = build_command(payload.get("action"), payload.get("invoice"))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self._send(
                400,
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json",
            )
            return

        result = run_command(command)
        self._send(200, json.dumps(result).encode("utf-8"), "application/json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    # Loopback only. This runs project commands as a subprocess, which is fine
    # on your own machine and is not something to expose on a network.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    print(f"\nInvoice pipeline UI: {url}")
    print("Press Ctrl-C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
