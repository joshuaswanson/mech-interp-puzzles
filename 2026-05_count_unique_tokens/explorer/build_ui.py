import json
from pathlib import Path

ui = Path(__file__).resolve().parent
template = (ui / "template.html").read_text()
css = (ui / "style.css").read_text()
model_js = (ui / "model.js").read_text()
app_js = (ui / "app.js").read_text()
weights = json.loads((ui / "model_weights.json").read_text())
model_json = json.dumps(weights, separators=(",", ":"))
assert "</script" not in model_json

html = (template
        .replace("/*__CSS__*/", css)
        .replace("/*__MODEL_JS__*/", model_js)
        .replace("/*__APP_JS__*/", app_js)
        .replace("__MODEL_JSON__", model_json))
out = ui.parent / "unique_count_explorer.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
