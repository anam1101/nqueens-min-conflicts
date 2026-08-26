# N-Queens Solver — Min-Conflicts Local Search

Places N non-attacking queens on an N×N board using the **min-conflicts** heuristic. Solves **n = 100,000** in under three minutes on a laptop — a problem size where classical backtracking is hopeless.

The interesting result: **the number of steps grows roughly linearly with n**, not exponentially.

| n | Steps | Time | Verified |
|---:|---:|---:|:--:|
| 8 | 30 | 0.004 s | ✅ |
| 100 | 190 | 0.006 s | ✅ |
| 1,000 | 713 | 0.038 s | ✅ |
| 10,000 | 6,215 | 1.83 s | ✅ |
| 100,000 | 61,147 | 142 s | ✅ |

*Reproduce with `--seed 42`; raw output in [`results/benchmark.json`](results/benchmark.json). Every solution is independently re-verified before being reported.*

Across four orders of magnitude, steps stay near **0.6 × n**. Backtracking on n = 100,000 would not finish in any useful amount of time.

## How it works

Instead of building a solution incrementally and undoing mistakes, min-conflicts starts from a **random complete assignment** and repeatedly repairs it: pick a queen that's attacked, move it to the row in its column where it conflicts with the fewest others, repeat.

Three design decisions make the scale possible:

**One queen per column, by construction.** The board is a 1-D array where `positions[col] = row`. Column conflicts become impossible, cutting the constraints to check by a third.

**O(1) moves.** Three count arrays track queens per row and per diagonal. Moving a queen updates exactly **six array elements** — independent of `n`. Without this, every move would be an O(n) rescan and n = 100,000 would be unreachable.

**Vectorised conflict scoring.** Evaluating all `n` candidate rows for a column is NumPy slice arithmetic over the diagonal arrays into a preallocated buffer, not a Python loop.

Random restarts handle the local minima that local search can get stuck in.

## Running it

```bash
pip install -r requirements.txt
```

Solve one instance and print the board:

```bash
python nqueens.py solve --n 8 --verify --show-board --seed 42
```

Benchmark across sizes:

```bash
python nqueens.py benchmark --n-values 10 1000 100000 --verify --output results/benchmark.json
```

`solve` also takes `--max-steps`, `--max-restarts`, and `--save-solution`.

## Tests

```bash
python -m pytest tests/ -q
```

15 tests covering solver correctness across sizes, seed reproducibility, rejection of the unsolvable n=2 and n=3 cases, and the validator itself — including that it catches shared rows, shared diagonals, and out-of-range values.

## Built with

Python 3 · NumPy. Type-hinted throughout, `dataclass` state, `argparse` CLI.
