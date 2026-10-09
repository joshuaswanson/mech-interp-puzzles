# 2026-10: Set Difference, Part 2

- [Puzzle page](https://puzzles.baulab.info/october-2026.html)
- Model on HuggingFace, [`andyrdt/10_2026_puzzle_1`](https://huggingface.co/andyrdt/10_2026_puzzle_1)
- [Official starter](https://github.com/andyrdt/puzzles/tree/main/10_2026)

**Task.** Given a set X of K distinct symbols (2 to 8, drawn from sixteen) and a permutation Y of it, read one symbol at a time and predict which symbols could come next. The model reads `[BOS] x1..xK [SEP] y1..yK [EOS]` and must output the uniform distribution over the symbols not yet consumed, then EOS. Two attention-only layers, one head each, `d_model = 64`, RMSNorm, rotary positions, 35,392 parameters.

**Answer.** The residual carries one number per symbol and that number is an occurrence count. Layer 0 is a counting head. It attends uniformly over every symbol token seen so far, flat to within a standard deviation of 0.0001, and its OV circuit is a negative copy, so each occurrence of a symbol subtracts a fixed amount from that symbol's logit. Layer 1 is a membership head parked on SEP with weight 1.000 at every step. The residual at SEP is layer 0's average over the X block, and layer 1's OV circuit is a positive copy, so it adds a fixed bonus to every symbol of X. The readout is membership minus count, and the two coefficients satisfy alpha = 2 beta at every K, so a symbol seen once stays positive and a symbol seen twice cancels to zero. The output is uniform over the survivors because the score depends on the count alone.

EOS works differently and is positional. The value at SEP is the only one in the sequence carrying a negative EOS component, so sitting on SEP suppresses EOS. Rotary position raises the competing keys as the query moves away from SEP while layer 0's average over the K symbols before SEP sets how high the SEP key starts, and the two cross at exactly distance K. The learned rule is "stop once as many symbols have been read after SEP as before it", which agrees with "stop when the set is empty" only because Y is always a permutation. Feeding Y sequences that repeat symbols separates the two rules and the model follows the length.

## Contents

- `starter_notebook.ipynb` is the official starter.
- `solution_notebook.ipynb` is the solution, executed.
- `analysis/` holds scripts `01` to `11` in the order they were run, and `common.py`.
