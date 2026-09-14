# 2026-05: Count Unique Tokens

- [Puzzle page](https://puzzles.baulab.info/may-2026.html)
- Model on HuggingFace, [`andyrdt/05_2026_puzzle_1`](https://huggingface.co/andyrdt/05_2026_puzzle_1)
- [Official starter](https://github.com/andyrdt/puzzles/tree/main/05_2026)

**Task.** Given ten symbols from an alphabet of ten, predict how many distinct symbols there are. Two attention layers, four heads each, `d_model = 32`, no positional embeddings.

**Answer.** The count is a sum of ten presence bits, one per symbol. In layer 0 each head is a priority encoder over its causal prefix. It attends to the highest-ranked symbol present under a fixed per-head ranking, which makes it a detector for its top symbol (`a`, `h`, `d`, `g` at ordinary positions, and `i`, `j`, `c`, `e` at the ANS position, whose query row has its own ranking). `b` and `f` queries flip or lose the ranking, which marks those positions so layer 1 can find them directly. At ANS, each layer-1 head adds one unit along a shared count direction per present symbol of its group. Head 0 counts `{b}`, head 1 counts `{f}`, head 2 counts `{a, d, g, h}` by copying the four layer-0 bits from a position whose prefix is complete, and head 3 counts `{c, e, i, j}` through how much attention stays on BOS. The unembedding turns U units along that direction into a parabola in the answer index that peaks at U.

## Contents

- `starter_notebook.ipynb` is the official starter.
- `solution_notebook.ipynb` is the solution, executed.
- `unique_count_explorer.html` is a single-file interactive page that runs the model in the browser. Type any input and see the attention matrix per head, the layer-0 key rankings, the layer-1 counters with a per-head walkthrough, and the quadratic readout. Equations render through KaTeX from a CDN. Everything else works offline.
- `analysis/` holds scripts `01` to `16` in the order they were run, and `common.py`.
- `explorer/` holds the sources for the page (`template.html`, `style.css`, `app.js`, `model.js`), `export_weights.py` (weights to JSON), `test_model.js` (checks the JavaScript forward pass against PyTorch), and `build_ui.py` (assembles the single file).
