"""
app_web.py - Interfaz Web Interactiva para Mini-Transformer
Servidor HTTP nativo de Python (sin dependencias adicionales de servidor web).
Abre automáticamente la interfaz en el navegador web local.
"""

import os
import sys
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

try:
    import torch
    from model import MiniTransformerLM
    from generate import load_model_and_vocab, generate_completion, select_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mini-Transformer Generativo</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #a855f7;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
      display: flex;
      justify-content: center;
    }
    .container {
      width: 100%;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    header h1 { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
    header p { color: var(--text-muted); font-size: 0.9rem; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: #0369a1;
      color: #e0f2fe;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .card h2 {
      font-size: 1.2rem;
      margin-bottom: 12px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    textarea, input[type="text"] {
      width: 100%;
      background: #0b1329;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--text);
      font-size: 1rem;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
      border-color: var(--primary);
    }
    .grid-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }
    .control-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--primary);
      cursor: pointer;
    }
    .btn-group {
      display: flex;
      gap: 12px;
      margin-top: 16px;
    }
    button {
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn-primary {
      background: var(--primary);
      color: #0f172a;
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--primary-hover);
      color: #ffffff;
    }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover:not(:disabled) {
      background: #475569;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .output-box {
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      font-family: ui-monospace, Menlo, Monaco, "Courier New", monospace;
      font-size: 0.95rem;
      color: #38bdf8;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 120px;
      max-height: 400px;
      overflow-y: auto;
    }
    .log-box {
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      font-family: ui-monospace, Menlo, monospace;
      font-size: 0.85rem;
      color: #a3e635;
      white-space: pre-wrap;
      max-height: 200px;
      overflow-y: auto;
    }
    .spinner {
      border: 2px solid rgba(255,255,255,0.2);
      border-left-color: currentColor;
      border-radius: 50%;
      width: 14px;
      height: 14px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Mini-Transformer Generativo</h1>
        <p>Modelo de Lenguaje Autorregresivo estilo GPT a nivel de caracteres</p>
      </div>
      <div>
        <span id="deviceBadge" class="badge">Iniciando...</span>
      </div>
    </header>

    <!-- Panel de Generación -->
    <div class="card">
      <h2>✨ Generación de Texto</h2>
      <label for="prompt">Texto inicial (Prompt):</label>
      <input type="text" id="prompt" value="En un lugar de la Mancha " placeholder="Escribe el inicio de la frase...">

      <div class="grid-controls">
        <div class="control-item">
          <div class="slider-header">
            <span>Temperatura</span>
            <span id="tempVal">0.75</span>
          </div>
          <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.75" oninput="document.getElementById('tempVal').innerText = this.value">
        </div>

        <div class="control-item">
          <div class="slider-header">
            <span>Top-K</span>
            <span id="topkVal">15</span>
          </div>
          <input type="range" id="topk" min="1" max="50" step="1" value="15" oninput="document.getElementById('topkVal').innerText = this.value">
        </div>

        <div class="control-item">
          <div class="slider-header">
            <span>Caracteres Nuevos</span>
            <span id="tokensVal">300</span>
          </div>
          <input type="range" id="tokens" min="50" max="800" step="25" value="300" oninput="document.getElementById('tokensVal').innerText = this.value">
        </div>
      </div>

      <div class="btn-group">
        <button id="genBtn" class="btn-primary" onclick="generateText()">Generar Texto</button>
      </div>

      <div style="margin-top: 16px;">
        <label>Resultado generado:</label>
        <div id="output" class="output-box">El texto generado aparecerá aquí...</div>
      </div>
    </div>

    <!-- Panel de Entrenamiento Rápido -->
    <div class="card">
      <h2>🚀 Entrenar / Reentrenar Modelo</h2>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 12px;">
        Entrena el Transformer con el dataset <code>data/input.txt</code> directamente desde el navegador.
      </p>

      <div class="btn-group">
        <button id="trainBtn" class="btn-secondary" onclick="startTraining()">Iniciar Entrenamiento (1000 iteraciones)</button>
      </div>

      <div style="margin-top: 16px;">
        <label>Registro del entrenamiento:</label>
        <div id="trainLog" class="log-box">Listo para entrenar.</div>
      </div>
    </div>
  </div>

  <script>
    async function checkStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('deviceBadge');
        if (!data.torch) {
          badge.innerText = 'PyTorch no instalado';
          badge.style.background = '#b91c1c';
          document.getElementById('genBtn').disabled = true;
          document.getElementById('trainBtn').disabled = true;
          document.getElementById('output').innerText = 'Por favor instala PyTorch ejecutando ./setup.sh en tu terminal.';
        } else {
          badge.innerText = `${data.device.toUpperCase()} | ${data.model_loaded ? 'Modelo Listo' : 'Sin Modelo'}`;
          badge.style.background = data.model_loaded ? '#15803d' : '#b45309';
          if (!data.model_loaded) {
            document.getElementById('output').innerText = 'El modelo aún no ha sido entrenado. Haz clic en "Iniciar Entrenamiento" abajo para comenzar.';
          }
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function generateText() {
      const prompt = document.getElementById('prompt').value;
      const temperature = parseFloat(document.getElementById('temp').value);
      const top_k = parseInt(document.getElementById('topk').value);
      const max_tokens = parseInt(document.getElementById('tokens').value);

      const btn = document.getElementById('genBtn');
      const out = document.getElementById('output');

      btn.disabled = true;
      btn.innerHTML = '<div class="spinner"></div> Generando...';
      out.innerText = 'Generando caracteres autorregresivamente...';

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, temperature, top_k, max_tokens })
        });
        const data = await res.json();
        if (data.error) {
          out.innerText = 'Error: ' + data.error;
        } else {
          out.innerText = data.result;
        }
      } catch (err) {
        out.innerText = 'Error de conexión: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Generar Texto';
      }
    }

    async function startTraining() {
      const btn = document.getElementById('trainBtn');
      const log = document.getElementById('trainLog');

      btn.disabled = true;
      btn.innerHTML = '<div class="spinner"></div> Entrenando...';
      log.innerText = '[*] Solicitando inicio de entrenamiento...\\n';

      try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        log.innerText = data.log || data.error;
        await checkStatus();
      } catch (err) {
        log.innerText += '\\nError: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Iniciar Entrenamiento (1000 iteraciones)';
      }
    }

    checkStatus();
  </script>
</body>
</html>
"""

# Estado global
server_state = {
    "model": None,
    "stoi": None,
    "itos": None,
    "device": None,
    "is_training": False,
    "training_log": ""
}

def init_model():
    if not TORCH_AVAILABLE:
        return
    server_state["device"] = select_device()
    model_path = "output/model.pt"
    vocab_path = "output/vocab.json"
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        try:
            m, s, it = load_model_and_vocab(model_path, vocab_path, server_state["device"])
            server_state["model"] = m
            server_state["stoi"] = s
            server_state["itos"] = it
            print(f"[*] Modelo cargado correctamente en {server_state['device']}")
        except Exception as e:
            print(f"[!] Error al cargar modelo existente: {e}")


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/status":
            self._send_json({
                "torch": TORCH_AVAILABLE,
                "device": str(server_state["device"]) if server_state["device"] else "none",
                "model_loaded": server_state["model"] is not None,
                "is_training": server_state["is_training"]
            })
        else:
            self.send_error(404, "Ruta no encontrada")

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        try:
            payload = json.loads(post_body) if post_body else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/generate":
            if not TORCH_AVAILABLE:
                self._send_json({"error": "PyTorch no está disponible. Ejecuta ./setup.sh"}, status=400)
                return
            if server_state["model"] is None:
                self._send_json({"error": "El modelo no está cargado. Primero debes entrenarlo."}, status=400)
                return

            prompt = payload.get("prompt", " ")
            temp = float(payload.get("temperature", 0.75))
            top_k = int(payload.get("top_k", 15))
            max_tokens = int(payload.get("max_tokens", 250))

            try:
                res = generate_completion(
                    server_state["model"],
                    server_state["stoi"],
                    server_state["itos"],
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_k=top_k,
                    device=server_state["device"]
                )
                self._send_json({"result": res})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif parsed.path == "/api/train":
            if not TORCH_AVAILABLE:
                self._send_json({"error": "PyTorch no está instalado. Ejecuta ./setup.sh"}, status=400)
                return
            if server_state["is_training"]:
                self._send_json({"error": "Ya hay un entrenamiento en progreso."}, status=400)
                return

            server_state["is_training"] = True
            log_messages = []

            try:
                with open("data/input.txt", "r", encoding="utf-8") as f:
                    text = f.read()

                chars = sorted(list(set(text)))
                vocab_size = len(chars)
                stoi = {ch: i for i, ch in enumerate(chars)}
                data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
                n = int(0.9 * len(data))
                train_data = data[:n]
                val_data = data[n:]

                block_size = 64
                batch_size = 32
                max_iters = 1000

                model = MiniTransformerLM(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    n_embd=128,
                    n_head=4,
                    n_layer=3,
                    dropout=0.1
                ).to(server_state["device"])

                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

                def get_batch(split):
                    d = train_data if split == "train" else val_data
                    ix = torch.randint(len(d) - block_size, (batch_size,))
                    x = torch.stack([d[i:i + block_size] for i in ix]).to(server_state["device"])
                    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]).to(server_state["device"])
                    return x, y

                log_messages.append(f"[*] Iniciando entrenamiento ({max_iters} iteraciones)...")
                for iter_idx in range(1, max_iters + 1):
                    xb, yb = get_batch("train")
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    if iter_idx % 200 == 0 or iter_idx == max_iters:
                        log_messages.append(f"Iteración {iter_idx:4d}/{max_iters} | Pérdida: {loss.item():.4f}")

                os.makedirs("output", exist_ok=True)
                vocab_meta = {
                    "chars": chars,
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embd": 128,
                    "n_head": 4,
                    "n_layer": 3,
                }
                with open("output/vocab.json", "w", encoding="utf-8") as f:
                    json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

                torch.save(model.state_dict(), "output/model.pt")
                log_messages.append("[+] ¡Entrenamiento completado y modelo guardado!")

                # Recargar modelo
                init_model()
                self._send_json({"log": "\\n".join(log_messages)})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            finally:
                server_state["is_training"] = False
        else:
            self.send_error(404, "Ruta no encontrada")


def main():
    init_model()
    port = 8000
    server = HTTPServer(("localhost", port), RequestHandler)
    url = f"http://localhost:{port}"
    print(f"[*] Servidor web iniciado en: {url}")
    print("[*] Abriendo interfaz en tu navegador...")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nServidor detenido.")


if __name__ == "__main__":
    main()
