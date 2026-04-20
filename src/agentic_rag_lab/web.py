from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AgenticRAG-Lab</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: linear-gradient(180deg, #09101f, #0d1430 45%, #0b1020); color: #edf2ff; }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 28px 18px 64px; }
    .hero { display: grid; gap: 10px; margin-bottom: 24px; }
    .hero h1 { margin: 0; font-size: 34px; }
    .hero p { margin: 0; color: #b8c1ec; line-height: 1.6; max-width: 900px; }
    .grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; align-items: start; }
    .stack { display: grid; gap: 20px; }
    .card { background: rgba(18, 25, 51, .95); border: 1px solid #24305e; border-radius: 18px; padding: 18px; box-shadow: 0 14px 40px rgba(0,0,0,.25); }
    .label { font-size: 12px; color: #9fb0ff; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
    .muted { color: #b8c1ec; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    input, textarea, select { width: 100%; border-radius: 12px; border: 1px solid #33437a; background: #0e1530; color: #edf2ff; padding: 12px 14px; }
    textarea { min-height: 120px; resize: vertical; }
    button { background: #5e72eb; color: white; border: 0; border-radius: 10px; padding: 12px 16px; font-weight: 600; cursor: pointer; }
    button.secondary { background: #1a2450; border: 1px solid #33437a; }
    button:hover { background: #7082f2; }
    button.secondary:hover { background: #24305e; }
    .pill { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #1a2450; border: 1px solid #33437a; margin-right: 8px; margin-bottom: 8px; font-size: 12px; color: #cfd8ff; }
    .answer { white-space: pre-wrap; line-height: 1.75; font-size: 15px; }
    .trace-item, .doc-item, .history-item, .result-item { padding: 12px; border-radius: 12px; background: #0e1530; border: 1px solid #273767; margin-bottom: 10px; }
    .example { display:block; width:100%; text-align:left; background:#0f1738; border:1px solid #2d3d72; border-radius: 12px; padding: 12px; color:#edf2ff; margin-bottom:10px; }
    pre { background:#0e1530; border-radius: 12px; padding: 12px; overflow:auto; border: 1px solid #273767; }
    .section-gap { margin-top: 18px; }
    .tiny { font-size: 12px; color: #95a6e8; }
    @media (max-width: 980px) { .grid, .grid-2 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AgenticRAG-Lab</h1>
      <p>A local Agentic RAG workbench: ask multi-hop questions, inspect retrieval, browse benchmark examples, and add your own knowledge-base documents without leaving the app.</p>
      <div id="stats" class="row muted"></div>
    </section>
    <section class="grid">
      <div class="stack">
        <div class="card">
          <div class="label">Ask the system</div>
          <textarea id="query" placeholder="Ask a HotpotQA question or something about your own uploaded documents..."></textarea>
          <div class="row section-gap">
            <button id="askBtn">Run Agentic RAG</button>
            <button id="inspectBtn" class="secondary">Inspect Retrieval</button>
            <select id="knowledgeBaseSelect" style="max-width: 220px;"></select>
            <select id="retrieveMethod" style="max-width: 180px;">
              <option value="hybrid">hybrid</option>
              <option value="semantic">semantic</option>
              <option value="lexical">lexical</option>
            </select>
            <span id="status" class="muted"></span>
          </div>
          <div class="section-gap">
            <div class="label">Answer</div>
            <div id="answer" class="answer muted">No query yet.</div>
          </div>
          <div class="grid-2 section-gap">
            <div>
              <div class="label">Reference Answer</div>
              <div id="reference" class="muted">Only shown when the query matches a known HotpotQA case.</div>
            </div>
            <div>
              <div class="label">Citations</div>
              <div id="citations" class="muted">No citations.</div>
            </div>
          </div>
          <div class="section-gap">
            <div class="label">Trace</div>
            <div id="trace"></div>
          </div>
        </div>

        <div class="card">
          <div class="label">Retrieval Inspector</div>
          <div class="tiny">See which documents the current retriever is using before or after you ask.</div>
          <div id="retrieveResults" class="section-gap muted">No retrieval run yet.</div>
        </div>
      </div>

      <div class="stack">
        <div class="card">
          <div class="label">Add custom knowledge</div>
          <input id="docTitle" placeholder="Document title" />
          <textarea id="docContent" class="section-gap" placeholder="Paste knowledge-base content here..."></textarea>
          <div class="row section-gap">
            <select id="docKnowledgeBase" style="max-width: 220px;"></select>
            <input id="docFile" type="file" accept=".txt,.md,.text" style="max-width: 280px;" />
            <button id="addDocBtn">Add Document</button>
          </div>
          <div class="row section-gap">
            <input id="kbName" placeholder="Create new knowledge base" style="max-width: 240px;" />
            <input id="kbDescription" placeholder="Optional description" />
            <button id="createKbBtn" class="secondary">Create KB</button>
          </div>
          <div id="docStatus" class="tiny section-gap">Documents are persisted locally and included in retrieval immediately.</div>
        </div>

        <div class="card">
          <div class="label">Knowledge Bases</div>
          <div id="knowledgeBases" class="muted">Loading knowledge bases...</div>
        </div>

        <div class="card">
          <div class="label">Knowledge Base</div>
          <div id="documents" class="muted">Loading documents...</div>
        </div>

        <div class="card">
          <div class="label">Recent History</div>
          <div id="history" class="muted">No questions asked yet.</div>
        </div>

        <div class="card">
          <div class="label">Example Questions</div>
          <div id="examples"></div>
        </div>

        <div class="card">
          <div class="label">Retrieval Ablation Snapshot</div>
          <pre id="benchmark">Loading...</pre>
        </div>
      </div>
    </section>
  </div>
  <script>
    async function getJson(url) {
      const response = await fetch(url);
      return await response.json();
    }
    async function postJson(url, body) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      return await response.json();
    }
    function renderStats(stats) {
      const node = document.getElementById('stats');
      node.innerHTML = `
        <span class="pill">${stats.case_count} benchmark cases</span>
        <span class="pill">${stats.corpus_document_count} total documents</span>
        <span class="pill">${stats.user_document_count} custom docs</span>
        <span class="pill">levels: ${Object.keys(stats.level_distribution).join(', ')}</span>
        <span class="pill">types: ${Object.keys(stats.type_distribution).join(', ')}</span>
      `;
    }
    function renderExamples(examples) {
      const node = document.getElementById('examples');
      node.innerHTML = '';
      for (const example of examples) {
        const button = document.createElement('button');
        button.className = 'example';
        button.textContent = `[${example.level}/${example.type}] ${example.question}`;
        button.onclick = () => { document.getElementById('query').value = example.question; runAsk(); };
        node.appendChild(button);
      }
    }
    function renderBenchmark(rows) {
      document.getElementById('benchmark').textContent = rows
        .map(row => `${row.method}@${row.top_k}  recall=${row.supporting_doc_recall.toFixed(3)}  all=${row.all_supporting_docs_hit_rate.toFixed(3)}  any=${row.any_supporting_doc_hit_rate.toFixed(3)}`)
        .join('\\n');
    }
    function renderKnowledgeBases(kbs) {
      const listNode = document.getElementById('knowledgeBases');
      listNode.innerHTML = kbs.map(kb => `
        <div class="doc-item">
          <div><strong>${kb.name}</strong> <span class="tiny">(${kb.source})</span></div>
          <div class="muted section-gap">${kb.description || 'No description.'}</div>
          <div class="tiny section-gap">${kb.document_count} documents</div>
        </div>
      `).join('');

      const selects = [document.getElementById('knowledgeBaseSelect'), document.getElementById('docKnowledgeBase')];
      for (const select of selects) {
        const current = select.value;
        select.innerHTML = '';
        for (const kb of kbs) {
          const option = document.createElement('option');
          option.value = kb.name;
          option.textContent = kb.name;
          select.appendChild(option);
        }
        if (current && [...select.options].some(option => option.value === current)) {
          select.value = current;
        } else if ([...select.options].some(option => option.value === 'all') && select.id === 'knowledgeBaseSelect') {
          select.value = 'all';
        } else if ([...select.options].some(option => option.value === 'workspace') && select.id === 'docKnowledgeBase') {
          select.value = 'workspace';
        }
      }
    }
    function renderAskResult(result) {
      document.getElementById('answer').textContent = result.answer || '(empty)';
      document.getElementById('reference').textContent = result.reference_answer || 'No exact reference match for this query.';
      document.getElementById('citations').innerHTML = result.citations.length
        ? result.citations.map(item => `<span class="pill">${item}</span>`).join('')
        : '<span class="muted">No citations.</span>';
      const traceNode = document.getElementById('trace');
      traceNode.innerHTML = '';
      for (const item of result.trace) {
        const div = document.createElement('div');
        div.className = 'trace-item';
        div.innerHTML = `<strong>${item.stage}</strong><div class="muted">${item.detail}</div>`;
        traceNode.appendChild(div);
      }
      document.getElementById('status').textContent = `route=${result.route} • iterations=${result.iterations}`;
    }
    function renderDocuments(documents) {
      const node = document.getElementById('documents');
      if (!documents.length) {
        node.innerHTML = '<span class="muted">No documents yet.</span>';
        return;
      }
      node.innerHTML = documents.map(doc => `
        <div class="doc-item">
          <div><strong>${doc.title}</strong> <span class="tiny">(${doc.source})</span></div>
          <div class="tiny">${doc.doc_id} • kb=${doc.knowledge_base}</div>
          <div class="muted section-gap">${doc.preview}</div>
        </div>
      `).join('');
    }
    function renderHistory(items) {
      const node = document.getElementById('history');
      if (!items.length) {
        node.innerHTML = '<span class="muted">No history yet.</span>';
        return;
      }
      node.innerHTML = items.map(item => `
        <div class="history-item">
          <div><strong>${item.query}</strong></div>
          <div class="tiny">route=${item.route}</div>
          <div class="muted section-gap">${item.answer}</div>
        </div>
      `).join('');
    }
    function renderRetrieveResults(payload) {
      const node = document.getElementById('retrieveResults');
      if (!payload.results || !payload.results.length) {
        node.innerHTML = '<span class="muted">No retrieval hits.</span>';
        return;
      }
      node.innerHTML = payload.results.map(item => `
        <div class="result-item">
          <div><strong>${item.title}</strong> <span class="tiny">(${item.source}) score=${item.score.toFixed(3)}</span></div>
          <div class="tiny">${item.doc_id}</div>
          <div class="muted section-gap">${item.preview}</div>
        </div>
      `).join('');
    }
    async function refreshSidePanels() {
      const selectedKb = document.getElementById('knowledgeBaseSelect').value || 'all';
      const [stats, docs, history, examples, benchmark, knowledgeBases] = await Promise.all([
        getJson('/api/stats'),
        getJson(`/api/documents?limit=10&knowledge_base=${encodeURIComponent(selectedKb)}`),
        getJson('/api/history?limit=6'),
        getJson('/api/examples?limit=8'),
        getJson('/api/benchmark'),
        getJson('/api/knowledge-bases')
      ]);
      renderStats(stats);
      renderKnowledgeBases(knowledgeBases.knowledge_bases);
      renderDocuments(docs.documents);
      renderHistory(history.history);
      renderExamples(examples.examples);
      renderBenchmark(benchmark.rows);
    }
    async function runAsk() {
      const query = document.getElementById('query').value.trim();
      if (!query) return;
      document.getElementById('status').textContent = 'Running...';
      const knowledge_base = document.getElementById('knowledgeBaseSelect').value || 'all';
      const result = await postJson('/api/ask', { query, knowledge_base });
      renderAskResult(result);
      await inspectRetrieval(query);
      await refreshSidePanels();
    }
    async function inspectRetrieval(explicitQuery) {
      const query = explicitQuery || document.getElementById('query').value.trim();
      if (!query) return;
      const method = document.getElementById('retrieveMethod').value;
      const knowledge_base = document.getElementById('knowledgeBaseSelect').value || 'all';
      const payload = await postJson('/api/retrieve', { query, top_k: 5, method, knowledge_base });
      renderRetrieveResults(payload);
    }
    async function addDocument() {
      const title = document.getElementById('docTitle').value.trim();
      const content = document.getElementById('docContent').value.trim();
      if (!title || !content) return;
      const knowledge_base = document.getElementById('docKnowledgeBase').value || 'workspace';
      document.getElementById('docStatus').textContent = 'Saving document...';
      const created = await postJson('/api/documents', { title, content, knowledge_base });
      document.getElementById('docStatus').textContent = `Saved ${created.title} to ${created.knowledge_base}`;
      document.getElementById('docTitle').value = '';
      document.getElementById('docContent').value = '';
      document.getElementById('docFile').value = '';
      await refreshSidePanels();
    }
    async function createKnowledgeBase() {
      const name = document.getElementById('kbName').value.trim();
      const description = document.getElementById('kbDescription').value.trim();
      if (!name) return;
      const created = await postJson('/api/knowledge-bases', { name, description });
      document.getElementById('docStatus').textContent = `Created knowledge base ${created.name}`;
      document.getElementById('kbName').value = '';
      document.getElementById('kbDescription').value = '';
      await refreshSidePanels();
      document.getElementById('docKnowledgeBase').value = created.name;
      document.getElementById('knowledgeBaseSelect').value = created.name;
    }
    document.getElementById('askBtn').addEventListener('click', runAsk);
    document.getElementById('inspectBtn').addEventListener('click', () => inspectRetrieval());
    document.getElementById('addDocBtn').addEventListener('click', addDocument);
    document.getElementById('createKbBtn').addEventListener('click', createKnowledgeBase);
    document.getElementById('knowledgeBaseSelect').addEventListener('change', refreshSidePanels);
    document.getElementById('docFile').addEventListener('change', async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      const text = await file.text();
      if (!document.getElementById('docTitle').value.trim()) {
        document.getElementById('docTitle').value = file.name;
      }
      document.getElementById('docContent').value = text;
      document.getElementById('docStatus').textContent = `Loaded file ${file.name}`;
    });
    refreshSidePanels();
  </script>
</body>
</html>
"""


def dispatch_request(service, *, method: str, path: str, body: bytes | None = None) -> tuple[int, str, bytes]:
    parsed = urlparse(path)

    if method == "GET":
        if parsed.path == "/":
            return HTTPStatus.OK, "text/html; charset=utf-8", HTML_PAGE.encode("utf-8")
        if parsed.path == "/api/health":
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps({"status": "ok"}).encode("utf-8")
        if parsed.path == "/api/stats":
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(service.stats(), ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/examples":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["8"])[0])
            payload = {"examples": service.example_questions(limit=limit)}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/benchmark":
            payload = {"rows": service.benchmark_summary()}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/documents":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["10"])[0])
            source = params.get("source", [None])[0]
            knowledge_base = params.get("knowledge_base", ["all"])[0]
            payload = {"documents": service.list_documents(limit=limit, source=source, knowledge_base=knowledge_base)}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/history":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["10"])[0])
            payload = {"history": service.history(limit=limit)}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/runs":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["20"])[0])
            payload = {"runs": service.runs(limit=limit)}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/knowledge-bases":
            payload = {"knowledge_bases": service.list_knowledge_bases()}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", json.dumps({"error": "Not found"}).encode("utf-8")

    if method == "POST":
        payload = json.loads(body or b"{}")
        if parsed.path == "/api/ask":
            query = str(payload.get("query", "")).strip()
            if not query:
                return HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", json.dumps({"error": "query is required"}).encode("utf-8")
            knowledge_base = str(payload.get("knowledge_base", "all"))
            result = service.ask(query, knowledge_base=knowledge_base)
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/documents":
            title = str(payload.get("title", "")).strip()
            content = str(payload.get("content", "")).strip()
            if not title or not content:
                return HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", json.dumps({"error": "title and content are required"}).encode("utf-8")
            knowledge_base = str(payload.get("knowledge_base", "workspace"))
            result = service.add_document(title=title, content=content, source="user", knowledge_base=knowledge_base)
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/knowledge-bases":
            name = str(payload.get("name", "")).strip()
            if not name:
                return HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", json.dumps({"error": "name is required"}).encode("utf-8")
            description = str(payload.get("description", "")).strip()
            result = service.create_knowledge_base(name, description=description)
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/retrieve":
            query = str(payload.get("query", "")).strip()
            if not query:
                return HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", json.dumps({"error": "query is required"}).encode("utf-8")
            top_k = int(payload.get("top_k", 5))
            method_name = str(payload.get("method", "hybrid"))
            knowledge_base = str(payload.get("knowledge_base", "all"))
            result = service.retrieve(query, top_k=top_k, method=method_name, knowledge_base=knowledge_base)
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8")
        if parsed.path == "/api/import-paths":
            paths = payload.get("paths", [])
            if not isinstance(paths, list) or not paths:
                return HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", json.dumps({"error": "paths is required"}).encode("utf-8")
            knowledge_base = str(payload.get("knowledge_base", "workspace"))
            result = {"documents": service.import_files(paths, knowledge_base=knowledge_base)}
            return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8")
        return HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", json.dumps({"error": "Not found"}).encode("utf-8")

    return HTTPStatus.METHOD_NOT_ALLOWED, "application/json; charset=utf-8", json.dumps({"error": "Method not allowed"}).encode("utf-8")


def create_server(service, *, host: str = "127.0.0.1", port: int = 8000):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = dispatch_request(service, method="GET", path=self.path)
            self._send(status, content_type, body)

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            status, content_type, body = dispatch_request(
                service,
                method="POST",
                path=self.path,
                body=self.rfile.read(content_length) if content_length else b"",
            )
            self._send(status, content_type, body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
