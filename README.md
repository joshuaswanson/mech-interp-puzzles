# Mech interp puzzles

Solutions to the [Mech Interp Puzzles](https://puzzles.baulab.info/) by andyrdt, a monthly series inspired by Callum McDougall's ARENA Monthly Algorithmic Challenges. Each puzzle is a tiny attention-only transformer trained to 100% accuracy on a toy task. The goal is to reverse-engineer the algorithm it learned.

| Puzzle | Task | Model | Solution |
|---|---|---|---|
| [2026-04](2026-04_max_of_list/) | max of a list of numbers | 1a has 1 layer on digits 0 to 9. 1b has 2 layers on two-digit numbers as digit tokens | [notebook](2026-04_max_of_list/solution_notebook.ipynb) |
| [2026-05](2026-05_count_unique_tokens/) | number of distinct symbols in a sequence of 10 | 2 layers, 4 heads, no positional embeddings | [notebook](2026-05_count_unique_tokens/solution_notebook.ipynb), [interactive explorer](2026-05_count_unique_tokens/unique_count_explorer.html) |
| [2026-09](2026-09_set_difference/) | which symbol was removed from a four-element set | 1 layer, 2 heads, `d_model = 2`, 108 parameters | [notebook](2026-09_set_difference/solution_notebook.ipynb) |

## Layout

Every puzzle directory has the same structure.

```
YYYY-MM_task/
  README.md                 task, model, links, one-paragraph answer
  starter_notebook.ipynb    the official starter (from github.com/andyrdt/puzzles)
  solution_notebook.ipynb   the solution, executed, with all outputs
  analysis/
    common.py               model loading and helpers
    NN_*.py                 exploratory scripts, in the order they were run
```

The May puzzle also has `explorer/`, the source of a single-file interactive page that runs the model in the browser.

## Running

Python dependencies are managed with [uv](https://docs.astral.sh/uv/). The models are downloaded from HuggingFace on first use.

```bash
uv sync
uv run 2026-09_set_difference/analysis/01_weights.py          # any exploratory script
uv run jupyter notebook                                       # open the solution notebooks
```

For the May explorer, `uv run 2026-05_count_unique_tokens/explorer/export_weights.py` writes the model weights to JSON. `node 2026-05_count_unique_tokens/explorer/test_model.js` checks the JavaScript forward pass against PyTorch. `uv run 2026-05_count_unique_tokens/explorer/build_ui.py` assembles `unique_count_explorer.html`.
