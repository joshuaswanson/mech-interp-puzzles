# 2026-09: Set Difference

- [Puzzle page](https://puzzles.baulab.info/september-2026.html)
- Model on HuggingFace, [`andyrdt/09_2026_puzzle_1`](https://huggingface.co/andyrdt/09_2026_puzzle_1)
- [Official starter](https://github.com/andyrdt/puzzles/tree/main/09_2026)

**Task.** Given a set X of four symbols (from sixteen) and X with one symbol z removed, both halves shuffled, predict z. One attention layer, two heads with `d_head = 1`, `d_model = 2`, learned positions, 108 parameters, correct on all 1,048,320 valid prompts.

**Answer.** The final residual is a 2-D vector whose direction depends only on z. The sixteen unembedding vectors form a fan in a fixed symbol order and read the direction off. The direction arises for four reasons. First, positional embeddings are identical within each half, so the model sees two bags. Second, moving a symbol from the X half to the Y half multiplies its attention weight by a fixed factor and shifts its value by a fixed amount. Third, the sixteen symbol embeddings lie on exactly the curve where an X copy plus a Y copy of any symbol contribute the same constant to a head's unnormalised sum, so repeated symbols become invisible and only the lone symbol z contributes something symbol-specific. Fourth, the two heads' normalisers stay in a fixed ratio, so the companions only rescale the vector. Pushing one symbol off the curve breaks exactly the problems where that symbol is a companion.

## Contents

- `starter_notebook.ipynb` is the official starter.
- `solution_notebook.ipynb` is the solution, executed.
- `analysis/` holds scripts `01` to `07` in the order they were run, and `common.py`.
