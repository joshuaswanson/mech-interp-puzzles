# 2026-04: Max of List

- [Puzzle page](https://puzzles.baulab.info/april-2026.html)
- Models on HuggingFace, [`andyrdt/04_2026_puzzle_1a`](https://huggingface.co/andyrdt/04_2026_puzzle_1a) and [`andyrdt/04_2026_puzzle_1b`](https://huggingface.co/andyrdt/04_2026_puzzle_1b)
- [Official starter](https://github.com/andyrdt/puzzles/tree/main/04_2026)

**Task.** Given five numbers, predict the maximum. Model 1a takes numbers 0 to 9 as one token each and has one attention layer. Model 1b takes numbers 0 to 99 as two digit tokens each and has two attention layers. It emits the tens digit, then the ones digit.

**Answer, model 1a.** Head 3's attention score from ANS to a number token rises with the number's value (4 to 9 nats per unit), so the softmax lands on copies of the maximum. Every head's OV path into the answer logits is rank 1, so each head writes one fixed logit direction scaled by the attended token. For max 2 to 6, three heads sit on ANS as fixed biases and head 3's growing scale walks the argmax up the ramp. For 7 to 9, heads 2 (attends to values of 7 or more) and 0 (attends to 9) add the separation head 3 cannot.

**Answer, model 1b.** The tens digit comes from layer-0 head 1, which attends to the largest tens-digit token, as in 1a. The ones digit takes both layers. Layer-0 head 3 at every ones-digit position attends to the tens digit just before it and copies it, so each ones position then holds the whole number. At the output position, layer-1 heads 0 and 3 score every ones position by about 16 x (stored tens) + 1.8 x (ones), attend to the largest, and copy its ones digit out. A one-layer model cannot do this because at the output position it cannot see which tens digit a ones digit belongs to.

## Contents

- `starter_notebook.ipynb` is the official starter.
- `solution_notebook.ipynb` is the solution, executed, covering both models.
- `analysis/` holds scripts `01a` to `03a` for model 1a, `01b` to `05b` for model 1b, and `common.py`.
