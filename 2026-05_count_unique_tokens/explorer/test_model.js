const fs = require("fs");
const { forward } = require("./model.js");
const model = JSON.parse(fs.readFileSync(__dirname + "/model_weights.json"));
const ref = JSON.parse(fs.readFileSync(__dirname + "/reference_outputs.json"));
let maxLogit = 0, maxAttn = 0;
for (const r of ref) {
  const out = forward(model, r.tokens);
  for (let c = 0; c < 10; c++) maxLogit = Math.max(maxLogit, Math.abs(out.logits[c] - r.logits[c]));
  for (let h = 0; h < 4; h++) {
    const w = out.layers[1].heads[h].weights[11];
    for (let j = 0; j < 12; j++) maxAttn = Math.max(maxAttn, Math.abs(w[j] - r.attn_l1_ans[h][j]));
    const w0 = out.layers[0].heads[h].weights[5];
    for (let j = 0; j <= 5; j++) maxAttn = Math.max(maxAttn, Math.abs(w0[j] - r.attn_l0_row5[h][j]));
  }
}
console.log("max |logit diff| vs PyTorch:", maxLogit.toExponential(2));
console.log("max |attention diff| vs PyTorch:", maxAttn.toExponential(2));
if (maxLogit > 1e-2 || maxAttn > 1e-4) { console.log("MISMATCH"); process.exit(1); }
console.log("JS forward pass matches PyTorch");
