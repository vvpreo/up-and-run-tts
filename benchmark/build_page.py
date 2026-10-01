"""Build the static listening page (.tmp/site) from generated samples, timings and ASR round-trip results."""
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".tmp/out"
SITE = ROOT / ".tmp/site"
TEXTS = json.loads((ROOT / "benchmark/texts.json").read_text())
EVAL = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else {}
CATALOG = json.loads((ROOT / "benchmark/catalog.json").read_text())  # display names, order, notes

SITE.mkdir(parents=True, exist_ok=True)
(SITE / "audio").mkdir(exist_ok=True)
metrics = {p.parent.name: json.loads(p.read_text()) for p in OUT.glob("*/metrics.json")}


def e(s):
    return html.escape(str(s))


rows = []  # (model dir, variant, display label, group)
for m in CATALOG["models"]:
    for v in m["variants"]:
        if (OUT / m["dir"] / v["id"]).is_dir():
            rows.append((m, v))


def normalize(src, dst, target_rms=0.1, peak=0.97):
    """Equal loudness for every sample: louder otherwise reads as 'better' when comparing by ear."""
    import numpy as np
    import soundfile as sf

    a, sr = sf.read(src, dtype="float32")
    rms = float(np.sqrt((a ** 2).mean())) or 1.0
    a = a * (target_rms / rms)
    top = float(np.abs(a).max())
    if top > peak:
        a = a * (peak / top)
    sf.write(dst, a, sr, subtype="PCM_16")


def cell_audio(m, v, tid):
    src = OUT / m["dir"] / v["id"] / f"{tid}.wav"
    if not src.exists():
        return None
    dst = SITE / "audio" / f"{m['dir']}__{v['id']}__{tid}.wav"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        normalize(src, dst)
    return f"audio/{dst.name}"


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


CER_SKIP = {"t11"}  # English terms: the ASR reference is ambiguous there
summary = []
for m, v in rows:
    runs = metrics.get(m["dir"], {}).get("runs", {})
    r = [runs[f"{v['id']}/{t['id']}"] for t in TEXTS if f"{v['id']}/{t['id']}" in runs and "rtf" in runs[f"{v['id']}/{t['id']}"]]
    cer = avg([EVAL.get(f"{m['dir']}/{v['id']}/{t['id']}", {}).get("cer") for t in TEXTS if t["id"] not in CER_SKIP])
    summary.append({"m": m, "v": v, "rtf": avg([x["rtf"] for x in r]), "ttfa": avg([x.get("ttfa_s") for x in r]), "cer": cer, "mos": avg([EVAL.get(f"{m['dir']}/{v['id']}/{t['id']}", {}).get("utmos") for t in TEXTS]),
                    "mem": metrics.get(m["dir"], {}).get("host_mem_used_gb"), "cuda": metrics.get(m["dir"], {}).get("cuda_peak_reserved_gb")})


def fmt(x, nd=2, suffix=""):
    return "—" if x is None else f"{x:.{nd}f}{suffix}"


P = []
P.append("""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTS: образцы на слух</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--line:#dcdcd6;--card:#fff;--acc:#2d5d8a;--good:#2e7d32;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#e8e8e3;--mut:#9a9a93;--line:#33332f;--card:#1f1f1c;--acc:#8ab4d8;--good:#81c784;--bad:#ef9a9a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:24px;margin:8px 0}h2{font-size:18px;margin:28px 0 6px}
p.lead{color:var(--mut);margin:4px 0 12px}.txt{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:6px 0}
.exp{color:var(--mut);font-size:13px}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle;font-size:14px}th{color:var(--mut);font-weight:600;font-size:12px}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}audio{height:32px;width:260px;max-width:100%}.hyp{color:var(--mut);font-size:12px;max-width:340px}
.wrap{overflow-x:auto}.grp td{background:var(--bg);font-weight:600;font-size:13px}.rate button{border:1px solid var(--line);background:none;color:var(--fg);border-radius:4px;width:26px;height:26px;cursor:pointer;padding:0}
.rate button.on{background:var(--acc);color:#fff;border-color:var(--acc)}nav a{color:var(--acc);margin-right:10px;font-size:13px;white-space:nowrap}
details.phrase{margin:14px 0;border:1px solid var(--line);border-radius:8px;background:var(--card)}
details.phrase>summary{cursor:pointer;padding:10px 12px;list-style:none;display:flex;gap:10px;align-items:baseline}
details.phrase>summary::before{content:'▸';color:var(--mut);flex:none}details.phrase[open]>summary::before{content:'▾'}
details.phrase>summary .sum{color:var(--mut);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
details.phrase[open]>summary .sum{display:none}details.phrase .txt,details.phrase .wrap{margin:0 12px 12px}
nav button{border:1px solid var(--line);background:none;color:var(--acc);border-radius:4px;cursor:pointer;font-size:12px}
.filter{margin:10px 0;display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:13px}
.filter label{white-space:nowrap;cursor:pointer}.filter button{border:1px solid var(--line);background:none;color:var(--acc);border-radius:4px;cursor:pointer;font-size:12px}
tr.hid{display:none}
.bad{color:var(--bad)}.good{color:var(--good)}#scores td{font-variant-numeric:tabular-nums}.note{color:var(--mut);font-size:13px}
</style></head><body><main>""")
P.append(f"<h1>Русский TTS: образцы для прослушивания</h1><p class='lead'>{e(CATALOG['lead'])}</p>")
REFS = json.loads((ROOT / "benchmark/refs/refs.json").read_text())
P.append("<h2>Исходные записи голосов</h2><div class='wrap'><table><tr><th>Голос</th><th>Запись, с которой снят клон</th><th>Текст записи</th></tr>")
for name, label in (("natasha", "Наташа"), ("ruslan", "Руслан")):
    normalize(ROOT / REFS[name]["wav"], SITE / "audio" / f"ref_{name}.wav")
    P.append(f"<tr><td>{label}</td><td><audio controls preload='none' src='audio/ref_{name}.wav'></audio></td><td class='hyp'>{e(REFS[name]['text'])}</td></tr>")
P.append("</table></div>")
P.append("<nav>" + "".join(f"<a href='#{t['id']}'>{e(t['cat'])}</a>" for t in TEXTS) + "<a href='#scores'>Мои оценки</a>"
         " <button id='openall'>раскрыть все</button> <button id='closeall'>свернуть все</button></nav>")
shown = [m for m in CATALOG["models"] if any((OUT / m["dir"] / v["id"]).is_dir() for v in m["variants"])]
P.append("<div class='filter'><span class='note'>Показывать модели:</span> "
         + "".join(f"<label><input type='checkbox' data-model='{e(m['dir'])}' checked> {e(m['name'])}</label>" for m in shown)
         + " <button id='fall'>все</button> <button id='fnone'>ни одной</button></div>")

P.append("<h2>Сводка</h2><div class='wrap'><table><tr><th>Модель</th><th>Вариант</th><th>Ошибки распознавания, %</th><th>Автооценка звучания (1–5)</th><th>Время синтеза / длительность</th><th>До первого звука, с</th><th>Память, ГБ</th></tr>")
last = None
for s in summary:
    name = s["m"]["name"] if s["m"]["dir"] != last else ""
    mem = fmt(s["cuda"], 1) if s["m"]["dir"] != last else ""
    last = s["m"]["dir"]
    P.append(f"<tr><td>{e(name)}</td><td>{e(s['v']['label'])}</td><td class='n'>{fmt(None if s['cer'] is None else s['cer'] * 100, 1)}</td>"
             f"<td class='n'>{fmt(s['mos'])}</td><td class='n'>{fmt(s['rtf'])}</td><td class='n'>{fmt(s['ttfa'])}</td><td class='n'>{mem}</td></tr>")
P.append("</table></div><p class='note'>" + e(CATALOG["summary_note"]) + "</p>")

for t in TEXTS:
    P.append(f"<details class='phrase' id='{t['id']}'><summary><b>{e(t['cat'])}</b> <span class='sum'>{e(t['text'])}</span></summary>"
             f"<div class='txt'>{e(t['text'])}"
             + (f"<div class='exp'>Правильно: {e(t['expect'])}</div>" if "expect" in t else "") + "</div>")
    P.append("<div class='wrap'><table><tr><th>Вариант</th><th>Звук</th><th>Оценка</th><th>Что услышал распознаватель</th></tr>")
    last = None
    for m, v in rows:
        a = cell_audio(m, v, t["id"])
        if not a:
            continue
        if m["dir"] != last:
            P.append(f"<tr class='grp' data-model='{e(m['dir'])}'><td colspan='4'>{e(m['name'])}</td></tr>")
            last = m["dir"]
        ev = EVAL.get(f"{m['dir']}/{v['id']}/{t['id']}", {})
        key = f"{m['dir']}/{v['id']}/{t['id']}"
        cer = ev.get("cer")
        cls = "" if cer is None or t["id"] in CER_SKIP else ("good" if cer <= 0.02 else "bad" if cer > 0.1 else "")
        P.append(f"<tr data-model='{e(m['dir'])}'><td>{e(v['label'])}</td><td><audio controls preload='none' src='{a}'></audio></td>"
                 f"<td class='rate' data-k='{e(key)}' data-m='{e(m['name'] + ' — ' + v['label'])}'></td>"
                 f"<td class='hyp'><span class='{cls}'>{fmt(None if cer is None else cer * 100, 1, '%')}</span> {e(ev.get('hyp', ''))}</td></tr>")
    P.append("</table></div></details>")

P.append("""<h2 id="scores">Мои оценки</h2><p class="note">Оценки от 1 до 5 сохраняются в этом браузере. Средняя по каждому варианту считается здесь.</p>
<div class="wrap"><table id="scores"></table></div><p><button id="copy">Скопировать оценки</button></p>
<script>
const KEY='tts-ratings-v1';let R={};try{R=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(R))}catch(e){}}
function draw(){const agg={};document.querySelectorAll('td.rate').forEach(td=>{const k=td.dataset.k,m=td.dataset.m;
 td.querySelectorAll('button').forEach(b=>b.classList.toggle('on',R[k]==+b.textContent));
 if(R[k]){(agg[m]=agg[m]||[]).push(R[k])}});
 const rows=Object.entries(agg).map(([m,a])=>[m,a.reduce((x,y)=>x+y,0)/a.length,a.length]).sort((a,b)=>b[1]-a[1]);
 document.getElementById('scores').innerHTML='<tr><th>Вариант</th><th>Средняя</th><th>Оценок</th></tr>'+rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1].toFixed(2)}</td><td>${r[2]}</td></tr>`).join('');
 return rows}
document.querySelectorAll('td.rate').forEach(td=>{for(let i=1;i<=5;i++){const b=document.createElement('button');b.textContent=i;
 b.onclick=()=>{const k=td.dataset.k;if(R[k]==i)delete R[k];else R[k]=i;save();draw()};td.appendChild(b);td.append(' ')}});
document.getElementById('copy').onclick=()=>{const t=draw().map(r=>`${r[0]}\\t${r[1].toFixed(2)}\\t${r[2]}`).join('\\n');navigator.clipboard&&navigator.clipboard.writeText(t)};
const FK='tts-models-v1';let F=null;try{F=JSON.parse(localStorage.getItem(FK))}catch(e){}
const boxes=[...document.querySelectorAll('.filter input')];
if(F)boxes.forEach(b=>b.checked=F.includes(b.dataset.model));
function filt(){const on=new Set(boxes.filter(b=>b.checked).map(b=>b.dataset.model));
 document.querySelectorAll('tr[data-model]').forEach(tr=>tr.classList.toggle('hid',!on.has(tr.dataset.model)));
 try{localStorage.setItem(FK,JSON.stringify([...on]))}catch(e){}}
boxes.forEach(b=>b.onchange=filt);
document.getElementById('fall').onclick=()=>{boxes.forEach(b=>b.checked=true);filt()};
document.getElementById('fnone').onclick=()=>{boxes.forEach(b=>b.checked=false);filt()};
filt();
document.getElementById('openall').onclick=()=>document.querySelectorAll('details.phrase').forEach(d=>d.open=true);
document.getElementById('closeall').onclick=()=>document.querySelectorAll('details.phrase').forEach(d=>d.open=false);
document.querySelectorAll('nav a[href^="#t"]').forEach(a=>a.onclick=()=>{const d=document.querySelector(a.getAttribute('href'));if(d)d.open=true});
if(location.hash){const d=document.querySelector(location.hash);if(d&&d.tagName=='DETAILS')d.open=true}
document.addEventListener('play',ev=>{document.querySelectorAll('audio').forEach(a=>{if(a!==ev.target)a.pause()})},true);draw();
</script></main></body></html>""")
(SITE / "index.html").write_text("\n".join(P))
print("rows", len(rows), "page", (SITE / "index.html").stat().st_size)
