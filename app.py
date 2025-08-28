# app.py
# PHPAiModel-GRU — GPU Trainer (Flask + SSE)
# - Character-level GRU, веса совместимы с aicore.php (Wz, Wr, Wh, Wy, bz, br, bh, by)
# - UI (/) и SSE-тренировка (/train)
# - GPU через CuPy (авто-фоллбек на NumPy)

import os, json, time, math, unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ---- Try GPU (CuPy) first, else fallback to NumPy ----
USE_GPU = True
try:
    import cupy as xp
    _ = xp.zeros((1,1))
    BACKEND = "GPU (CuPy/CUDA)"
except Exception:
    import numpy as xp  # type: ignore
    USE_GPU = False
    BACKEND = "CPU (NumPy)"

from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "Datasets"
MODELS_DIR   = BASE_DIR / "Models"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HTML = """<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GRU Trainer — PHPAiModel-GRU</title>
<style>
:root { --bg:#f9fafc; --panel:#ffffff; --ink:#1e1e2e; --muted:#667085; --acc:#3b82f6; }
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,Arial}
.wrap{max-width:880px;margin:0 auto;padding:24px}
h1{margin:8px 0 16px;font-size:24px}
.card{background:var(--panel);border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 4px 12px rgba(0,0,0,.05);margin-bottom:16px}
.card h2{font-size:18px;margin:0;padding:16px;border-bottom:1px solid #f0f0f0}
.card .body{padding:16px}
.muted{color:var(--muted)}
.row2{display:grid;gap:10px;grid-template-columns:1fr 1fr 1fr}
label{display:block;margin:8px 0 6px}
select,input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #cbd5e1;background:white;color:var(--ink)}
button{cursor:pointer;border:1px solid #cbd5e1;background:#f9fafc;color:var(--ink);padding:10px 14px;border-radius:12px}
button.primary{background:var(--acc);color:white;border:none}
.progress{height:10px;background:#f3f4f6;border-radius:999px;overflow:hidden;border:1px solid #e5e7eb}
.bar{height:100%;width:0;background:linear-gradient(90deg,#3b82f6,#60a5fa)}
.log{font-family:ui-monospace,Consolas,monospace;font-size:12px;height:300px;overflow:auto;background:#fdfdfd;border:1px solid #e5e7eb;border-radius:10px;padding:10px;white-space:pre;line-height:1.45}
.row{display:flex;gap:8px;flex-wrap:wrap}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin-bottom:8px}
</style></head>
<body>
<div class="wrap">
  <h1>GRU Trainer <span class="muted">· PHPAiModel-GRU</span></h1>

  <div class="card">
    <h2>Система</h2>
    <div class="body">
      <div class="kv">
        <div class="muted">Backend:</div><div id="backend">...</div>
        <div class="muted">Datasets dir:</div><div id="ddir">/Datasets</div>
        <div class="muted">Models dir:</div><div id="mdir">/Models</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Параметры обучения</h2>
    <div class="body">
      <label>Датасеты (*.txt)</label>
      <select id="datasets" multiple size="7"></select>

      <div class="row2">
        <div><label>Hidden size (H)</label><input id="H" type="number" min="16" max="512" step="16" value="256"></div>
        <div><label>Seq len</label><input id="SEQ" type="number" min="16" max="256" step="16" value="128"></div>
        <div><label>Epochs</label><input id="EPOCHS" type="number" min="1" max="100" value="15"></div>
      </div>

      <div class="row2" style="margin-top:8px">
        <div><label>Learning rate</label><input id="LR" type="number" step="0.001" min="0.0001" max="1" value="0.01"></div>
        <div><label>Output name (optional)</label><input id="OUT" placeholder="gru_ruen_H64.json"></div>
        <div style="display:flex;align-items:flex-end;gap:8px">
          <button class="primary" id="train">Start training</button>
        </div>
      </div>

      <div style="margin-top:12px" class="progress"><div id="bar" class="bar"></div></div>
      <div id="status" class="muted" style="margin-top:6px">Idle</div>
      <div id="log" class="log"></div>
    </div>
  </div>
</div>

<script>
const elDatasets = document.getElementById('datasets');
const elH = document.getElementById('H');
const elSEQ = document.getElementById('SEQ');
const elEPOCHS = document.getElementById('EPOCHS');
const elLR = document.getElementById('LR');
const elOUT = document.getElementById('OUT');
const elBar = document.getElementById('bar');
const elLog = document.getElementById('log');
const elStatus = document.getElementById('status');

document.getElementById('backend').textContent = "{{backend}}";

function appendLog(line){ elLog.textContent += line + "\\n"; elLog.scrollTop = elLog.scrollHeight; }

fetch('/list').then(r=>r.json()).then(j=>{
  (j.files||[]).forEach(f=>{
    const o = document.createElement('option');
    o.value = f; o.textContent = f; elDatasets.appendChild(o);
  });
});

let es;
function startTraining(){
  if (es) es.close();
  elLog.textContent=''; elBar.style.width='0%'; elStatus.textContent='Starting…';

  const ds = Array.from(elDatasets.selectedOptions).map(o=>o.value);
  if (ds.length===0){ alert('Выберите хотя бы один датасет'); return; }

  const params = new URLSearchParams({
    H:elH.value, SEQ:elSEQ.value, EPOCHS:elEPOCHS.value, LR:elLR.value, OUT:elOUT.value
  });
  ds.forEach(d=>params.append('dataset[]', d));

  es = new EventSource('/train?' + params.toString());
  es.addEventListener('progress', e=>{
    const j = JSON.parse(e.data);
    if (j.percent != null) elBar.style.width = j.percent.toFixed(2) + '%';
    if (j.msg) elStatus.textContent = j.msg;
    if (j.note) appendLog(j.note);
  });
  es.addEventListener('done', e=>{
    const j = JSON.parse(e.data);
    elBar.style.width = '100%';
    elStatus.textContent = 'Saved: ' + j.out_file;
    if (j.header) appendLog(j.header);
    if (j.footer) appendLog(j.footer);
    appendLog('Model saved to: ' + j.out_path);
    es.close();
  });
  es.addEventListener('error', ()=>{ elStatus.textContent = 'Stream closed'; });
}
document.getElementById('train').onclick = startTraining;
</script>
</body></html>
"""

def list_txt(dirpath: Path) -> List[str]:
    if not dirpath.is_dir(): return []
    return sorted([f.name for f in dirpath.iterdir() if f.is_file() and f.suffix.lower()=='.txt'])

def fmt_hms(sec: float) -> str:
    if sec < 0: sec = 0
    s = int(round(sec))
    h, m = divmod(s, 3600)
    m, ss = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"

# --------- Math helpers (xp supports cupy/numpy) ----------
def randn_like(shape, scale: float):
    # Box-Muller with xp for deterministic backend independence
    u1 = xp.clip(xp.random.random(shape), 1e-12, 1.0)
    u2 = xp.random.random(shape)
    R = xp.sqrt(-2.0 * xp.log(u1))
    theta = 2.0 * xp.pi * u2
    z = R * xp.cos(theta)
    return z * scale

def init_mat(rows: int, cols: int, scale: float):
    return randn_like((rows, cols), scale)

def zeros(n: int):
    return xp.zeros((n,), dtype=xp.float32)

def clip_(arr, c: float = 5.0):
    xp.clip(arr, -c, c, out=arr)

def sigmoid(x):
    return 1.0 / (1.0 + xp.exp(-x))

def softmax(logits):
    m = xp.max(logits)
    e = xp.exp(logits - m)
    s = xp.sum(e)
    if s <= 0:
        return xp.full_like(logits, 1.0 / logits.size)
    return e / s

# --------- GRU forward/backward (как в PHP раскладке) ----------
def step_forward(W, hprev, x_id, H, V, eyeV):
    # Concatenate [hprev, onehot(x)]
    x_one = eyeV[x_id]  # (V,)
    inp = xp.concatenate([hprev, x_one], axis=0)  # (H+V,)

    z = sigmoid(W['Wz'] @ inp + W['bz'])        # (H,)
    r = sigmoid(W['Wr'] @ inp + W['br'])        # (H,)
    u = r * hprev                                # elementwise
    # for Wh вход = [u, onehot(x)] по той же схеме:
    inp_h = xp.concatenate([u, x_one], axis=0)   # (H+V,)
    h_tilde = xp.tanh(W['Wh'] @ inp_h + W['bh']) # (H,)
    h = (1.0 - z) * hprev + z * h_tilde         # (H,)

    logits = W['Wy'] @ h + W['by']              # (V,)
    probs = softmax(logits)

    cache = (x_id, hprev, z, r, u, h_tilde, h)  # всё нужно для бэкварда
    return h, probs, cache

def backward_through_time(W, caches, targets, H, V, eyeV):
    # init grads
    d = {
        'Wz': xp.zeros_like(W['Wz']), 'Wr': xp.zeros_like(W['Wr']),
        'Wh': xp.zeros_like(W['Wh']), 'Wy': xp.zeros_like(W['Wy']),
        'bz': xp.zeros_like(W['bz']), 'br': xp.zeros_like(W['br']),
        'bh': xp.zeros_like(W['bh']), 'by': xp.zeros_like(W['by']),
    }
    dh_next = xp.zeros((H,), dtype=xp.float32)

    for t in range(len(caches)-1, -1, -1):
        x_id, h_prev, z, r, u, h_tilde, h_t = caches[t]
        y_id = targets[t]

        # dy
        logits_grad = xp.copy(softmax(W['Wy'] @ h_t + W['by']))
        logits_grad[y_id] -= 1.0  # dL/dlogits
        d['by'] += logits_grad
        d['Wy'] += xp.outer(logits_grad, h_t)

        # dh from output + next
        dh = (W['Wy'].T @ logits_grad) + dh_next

        # split gates
        dz_raw = dh * (h_tilde - h_prev)
        dh_tilde_raw = dh * z
        dh_prev = dh * (1.0 - z)

        dz_pre = dz_raw * z * (1.0 - z)                # sigmoid'
        dh_tilde_pre = dh_tilde_raw * (1.0 - h_tilde*h_tilde)  # tanh'

        # paths through Wh ([u; onehot])
        x_one = eyeV[x_id]
        inp_h = xp.concatenate([u, x_one], axis=0)
        d['bh'] += dh_tilde_pre
        d['Wh'] += xp.outer(dh_tilde_pre, inp_h)

        du = W['Wh'][:, :H].T @ dh_tilde_pre           # back to u
        dr_raw = du * h_prev
        dh_prev += du * r

        dr_pre = dr_raw * r * (1.0 - r)

        # z, r paths through Wz/Wr with inp = [h_prev; onehot]
        inp = xp.concatenate([h_prev, x_one], axis=0)
        d['bz'] += dz_pre
        d['br'] += dr_pre
        d['Wz'] += xp.outer(dz_pre, inp)
        d['Wr'] += xp.outer(dr_pre, inp)

        # back to h_prev from gates
        dh_prev += W['Wz'][:, :H].T @ dz_pre
        dh_prev += W['Wr'][:, :H].T @ dr_pre

        dh_next = dh_prev

    # clip
    for k in d:
        clip_(d[k], 5.0)
    return d

# ------------- Utils -------------
def utf8_chars(s: str) -> List[str]:
    # точный сплит по символам Юникода
    return [ch for ch in s]

def read_files(names: List[str]) -> str:
    buf = []
    for n in names:
        p = (DATASETS_DIR / Path(n).name)
        if p.exists() and p.is_file():
            buf.append(p.read_text(encoding='utf-8', errors='ignore'))
    return "".join(buf)

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

# ------------- Routes -------------
@app.get("/")
def index():
    return HTML.replace("{{backend}}", BACKEND)

@app.get("/list")
def list_datasets():
    return {"ok": True, "files": list_txt(DATASETS_DIR)}

@app.get("/train")
def train_sse():
    # headers for SSE
    def generate():
        H = int(max(16, min(512, int(request.args.get("H", 256)))))
        SEQ = int(max(16, min(256, int(request.args.get("SEQ", 128)))))
        EPOCHS = int(max(1, min(100, int(request.args.get("EPOCHS", 15)))))
        LR = float(request.args.get("LR", 0.01))
        OUT = (request.args.get("OUT") or "").strip()
        ds = request.args.getlist("dataset[]")

        if not ds:
            yield sse_event("progress", {"msg":"No datasets selected","note":"Ошибка: датасеты не выбраны","percent":0})
            return

        text = read_files(ds)
        if not text:
            yield sse_event("progress", {"msg":"Datasets empty","note":"Ошибка: файлы пустые или не найдены","percent":0})
            return
        text = text.replace("\r", "")

        # vocab
        char_list = utf8_chars(text)
        chars = {ch: True for ch in char_list}
        chars["\uFFFD"] = True
        ivocab = sorted(chars.keys())
        vocab = {ch:i for i,ch in enumerate(ivocab)}
        V = len(ivocab)

        # ids
        ids = xp.array([vocab.get(ch, vocab["\uFFFD"]) for ch in char_list], dtype=xp.int32)
        N = int(ids.shape[0])
        total_steps_per_epoch = max(1, (N-1)//SEQ)
        planned_total_steps = EPOCHS * total_steps_per_epoch

        # weights
        inZ = H + V
        inH = H + V
        scaleZ = 1.0 / math.sqrt(inZ)
        scaleH = 1.0 / math.sqrt(inH)
        scaleY = 1.0 / math.sqrt(H)

        W = {
            'Wz': init_mat(H, inZ, scaleZ).astype(xp.float32),
            'Wr': init_mat(H, inZ, scaleZ).astype(xp.float32),
            'Wh': init_mat(H, inH, scaleH).astype(xp.float32),
            'Wy': init_mat(V, H, scaleY).astype(xp.float32),
            'bz': zeros(H), 'br': zeros(H), 'bh': zeros(H), 'by': zeros(V)
        }

        # one-hot identity for quick indexing on GPU/CPU
        eyeV = xp.eye(V, dtype=xp.float32)

        header_lines = [
            "Запуск обучения…",
            f"Backend: {BACKEND}",
            "Dataset: " + ",".join(ds),
            f"Tokens: {N}",
            f"Vocab: {V}",
            f"H: {H}  SEQ: {SEQ}  Epochs: {EPOCHS}  LR: {LR}",
            f"Planned steps: {planned_total_steps} (≈ {total_steps_per_epoch} / epoch)"
        ]
        yield sse_event("progress", {"msg":"Training started","note":"\n".join(header_lines) + "\n"})

        start = time.time()
        seen_steps = 0
        loss_acc = 0.0
        tokens_acc = 0

        # state
        h = xp.zeros((H,), dtype=xp.float32)

        for epoch in range(1, EPOCHS+1):
            # simple sequential slicing, как в PHP
            i = 0
            for s in range(total_steps_per_epoch):
                xs = ids[i:i+SEQ]
                ys = ids[i+1:i+SEQ+1]
                T = int(ys.shape[0])
                i += SEQ

                caches = []
                # forward
                for t in range(T):
                    h, probs, cache = step_forward(W, h, int(xs[t]), H, V, eyeV)
                    caches.append(cache)
                    p = float(probs[int(ys[t])])
                    if p <= 1e-9: p = 1e-9
                    loss_acc += -math.log(p)
                    tokens_acc += 1

                # backward
                d = backward_through_time(W, caches, [int(k) for k in xp.asnumpy(ys) if not USE_GPU] if USE_GPU is False else [int(v) for v in ys.get()] , H, V, eyeV)

                # SGD
                for M in ['Wz','Wr','Wh']:
                    W[M] -= LR * d[M]
                W['Wy'] -= LR * d['Wy']
                W['bz'] -= LR * d['bz']; W['br'] -= LR * d['br']; W['bh'] -= LR * d['bh']; W['by'] -= LR * d['by']

                # progress
                seen_steps += 1
                if (seen_steps % 50) == 0 or seen_steps == 1:
                    pct = 100.0 * seen_steps / max(1, planned_total_steps)
                    spent = time.time() - start
                    eta = (planned_total_steps - seen_steps) * (spent / max(1, seen_steps))
                    avg = (loss_acc / max(1, tokens_acc))
                    line = "Progress: %7.2f%% | ETA %s | Spent %s | epoch %d/%d | step %d/%d | avg loss %.5f" % (
                        pct, fmt_hms(eta), fmt_hms(spent), epoch, EPOCHS, s+1, total_steps_per_epoch, avg
                    )
                    yield sse_event("progress", {
                        "percent": pct,
                        "msg": f"Epoch {epoch}/{EPOCHS} · step {s+1}/{total_steps_per_epoch}",
                        "note": line
                    })

        # перенос весов на CPU для JSON, если мы на GPU
        def to_cpu(a):
            try:
                return xp.asnumpy(a).tolist()
            except Exception:
                return a.tolist()

        meta = {
            "dataset_files": ds,
            "epochs": EPOCHS,
            "seq_len": SEQ,
            "lr": LR,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "stop_on_newline": True,
            "backend": BACKEND
        }
        out_name = OUT if OUT else f"gru_{(ds[0] if ds else 'dataset')}_H{H}_E{EPOCHS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_name = os.path.basename(out_name)
        out_path = MODELS_DIR / out_name

        model = {
            "H": H,
            "V": V,
            "vocab": {k:v for k,v in vocab.items()},
            "ivocab": ivocab,
            "W": {
                "Wz": to_cpu(W['Wz']),
                "Wr": to_cpu(W['Wr']),
                "Wh": to_cpu(W['Wh']),
                "Wy": to_cpu(W['Wy']),
                "bz": to_cpu(W['bz']),
                "br": to_cpu(W['br']),
                "bh": to_cpu(W['bh']),
                "by": to_cpu(W['by']),
            },
            "meta": meta
        }
        out_path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

        yield sse_event("done", {
            "ok": True,
            "out_file": out_name,
            "out_path": str(out_path),
            "header": "\n".join(header_lines),
            "footer": "Готово."
        })

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), headers=headers)

if __name__ == "__main__":
    # Запуск: python app.py
    # Открой http://127.0.0.1:5000/ — выбери .txt и жми Start training
    app.run(host="0.0.0.0", port=5000, threaded=True)
