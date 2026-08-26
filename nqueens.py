"""
N-Queens Solver using MIN-CONFLICTS Algorithm

This module implements the MIN-CONFLICTS local search algorithm to solve
the n-queens constraint satisfaction problem. The algorithm can efficiently
solve instances with up to millions of queens.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class SolveStats:
    """Statistics from a solve attempt."""

    n: int  # Board size
    solved: bool  # Whether solution was found
    steps: int  # Total iterations used
    restarts: int  # Number of random restarts
    duration_sec: float  # Wall-clock time in seconds
    max_steps: int  # Step limit per restart

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        payload = asdict(self)
        payload["duration_sec"] = round(self.duration_sec, 6)
        return payload


class MinConflictsSolver:
    """
    MIN-CONFLICTS solver for the n-queens problem.

    The solver uses a 1-D array representation where positions[col] = row,
    ensuring one queen per column by construction. Three auxiliary count
    arrays track queens per row and diagonal for O(1) conflict updates.
    """

    def __init__(
        self,
        n: int,
        max_steps: Optional[int] = None,
        max_restarts: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the solver.

        Args:
            n: Board size (number of queens)
            max_steps: Maximum iterations per restart (default: max(50, 4*n))
            max_restarts: Number of random restarts if stuck (default: 3)
            seed: Random seed for reproducibility
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n in (2, 3):
            raise ValueError(f"n={n} has no solutions")

        self.n = n
        self.max_steps = max_steps or max(50, 4 * n)
        self.max_restarts = max(1, max_restarts)
        self.rng = np.random.default_rng(seed)

        # Diagonal array length: 2n-1 diagonals in each direction
        self.diag_len = 2 * n - 1
        # Offset to make diagonal indices non-negative
        self.diag_offset = n - 1
        # Pre-computed column indices for vectorized operations
        self.cols = np.arange(n, dtype=np.int64)

    def solve(self) -> tuple[SolveStats, Optional[np.ndarray]]:
        """
        Run the MIN-CONFLICTS algorithm.

        Returns:
            Tuple of (statistics, solution array or None if failed)
        """
        start = time.perf_counter()
        # Pre-allocate buffer for value conflict computation
        buffer = np.empty(self.n, dtype=np.int64)

        for restart in range(self.max_restarts):
            # Initialize with random queen placement
            state = self._random_state()

            for step in range(self.max_steps):
                # Find all columns with conflicts
                conflicts = self._column_conflicts(state)
                conflict_cols = np.flatnonzero(conflicts > 0)

                # If no conflicts, we found a solution
                if conflict_cols.size == 0:
                    duration = time.perf_counter() - start
                    stats = SolveStats(
                        self.n, True, step, restart, duration, self.max_steps
                    )
                    return stats, state.positions.copy()

                # Pick a random conflicting column
                col = int(self.rng.choice(conflict_cols))

                # Find the row with minimum conflicts for this column
                self._value_conflicts(state, col, buffer)
                min_conflict = buffer.min()
                candidates = np.flatnonzero(buffer == min_conflict)
                new_row = int(self.rng.choice(candidates))  # Break ties randomly

                # Move queen if it changes position
                if new_row != state.positions[col]:
                    self._move(state, col, new_row)

        # Failed to find solution after all restarts
        duration = time.perf_counter() - start
        stats = SolveStats(
            self.n, False, self.max_steps, self.max_restarts, duration, self.max_steps
        )
        return stats, None

    def _random_state(self) -> "SolverState":
        """
        Generate a random initial state.

        Places one queen randomly in each column and computes
        the initial row and diagonal count arrays.
        """
        # Random row for each column
        positions = self.rng.integers(0, self.n, size=self.n, dtype=np.int64)

        # Count queens per row
        row_counts = np.bincount(positions, minlength=self.n).astype(np.int64)

        # Count queens per diagonal
        # diag1: down-right diagonal, index = row - col + offset
        diag1_indices = positions - self.cols + self.diag_offset
        diag1_counts = np.bincount(diag1_indices, minlength=self.diag_len).astype(
            np.int64
        )

        # diag2: down-left diagonal, index = row + col
        diag2_indices = positions + self.cols
        diag2_counts = np.bincount(diag2_indices, minlength=self.diag_len).astype(
            np.int64
        )

        return SolverState(positions, row_counts, diag1_counts, diag2_counts)

    def _column_conflicts(self, state: "SolverState") -> np.ndarray:
        """
        Compute the number of conflicts for each column's queen.

        A queen conflicts with others sharing its row or diagonals.
        We subtract 1 from each count to exclude the queen itself.

        Returns:
            Array of conflict counts, one per column
        """
        rows = state.positions

        # Row conflicts: count of queens in same row minus self
        row_conf = state.row_counts.take(rows) - 1

        # Diagonal conflicts
        diag1_idx = rows - self.cols + self.diag_offset
        diag2_idx = rows + self.cols
        diag1_conf = state.diag1_counts.take(diag1_idx) - 1
        diag2_conf = state.diag2_counts.take(diag2_idx) - 1

        return row_conf + diag1_conf + diag2_conf

    def _value_conflicts(
        self, state: "SolverState", column: int, buffer: np.ndarray
    ) -> None:
        """
        Compute conflict count for each possible row in a given column.

        This evaluates all n rows to find which has minimum conflicts.
        Results are stored in the provided buffer array.

        Args:
            state: Current board state
            column: Column to evaluate
            buffer: Pre-allocated array to store results
        """
        # Slice of diag1_counts corresponding to this column
        start1 = self.diag_offset - column
        start2 = column

        # Sum conflicts from both diagonals and rows
        np.copyto(buffer, state.diag1_counts[start1 : start1 + self.n])
        buffer += state.diag2_counts[start2 : start2 + self.n]
        buffer += state.row_counts

        # Subtract 3 from current position: the queen is counted in
        # row_counts, diag1_counts, and diag2_counts, but shouldn't
        # conflict with itself
        current_row = state.positions[column]
        buffer[current_row] -= 3

    def _move(self, state: "SolverState", column: int, new_row: int) -> None:
        """
        Move a queen to a new row and update all count arrays.

        This is O(1) - only updates 6 array elements regardless of n.

        Args:
            state: Current board state
            column: Column of queen to move
            new_row: Target row
        """
        old_row = state.positions[column]
        if old_row == new_row:
            return

        # Update position
        state.positions[column] = new_row

        # Update row counts
        state.row_counts[old_row] -= 1
        state.row_counts[new_row] += 1

        # Update diagonal counts
        old_d1 = old_row - column + self.diag_offset
        old_d2 = old_row + column
        new_d1 = new_row - column + self.diag_offset
        new_d2 = new_row + column

        state.diag1_counts[old_d1] -= 1
        state.diag1_counts[new_d1] += 1
        state.diag2_counts[old_d2] -= 1
        state.diag2_counts[new_d2] += 1


@dataclass
class SolverState:
    """
    Encapsulates the current board state.

    Attributes:
        positions: Array where positions[col] = row of queen in that column
        row_counts: Number of queens in each row
        diag1_counts: Number of queens on each down-right diagonal
        diag2_counts: Number of queens on each down-left diagonal
    """

    positions: np.ndarray
    row_counts: np.ndarray
    diag1_counts: np.ndarray
    diag2_counts: np.ndarray


def is_valid_solution(positions: Sequence[int]) -> bool:
    """
    Verify that a configuration is a valid n-queens solution.

    Checks that no two queens share a row or diagonal.
    Column uniqueness is guaranteed by the array representation.

    Args:
        positions: Array where positions[col] = row

    Returns:
        True if valid solution, False otherwise
    """
    n = len(positions)
    diag1 = set()  # Down-right diagonals (row - col)
    diag2 = set()  # Down-left diagonals (row + col)
    rows = set()  # Rows used

    for col, row in enumerate(positions):
        # Check row is in valid range
        if not 0 <= row < n:
            return False

        # Check row not already used
        if row in rows:
            return False

        # Check diagonals not already used
        key1 = row - col  # Down-right diagonal identifier
        key2 = row + col  # Down-left diagonal identifier
        if key1 in diag1 or key2 in diag2:
            return False

        # Record this queen's position
        rows.add(row)
        diag1.add(key1)
        diag2.add(key2)

    return True


def run_solve(args: argparse.Namespace) -> None:
    """CLI handler for solving a single instance."""
    solver = MinConflictsSolver(
        n=args.n,
        max_steps=args.max_steps,
        max_restarts=args.max_restarts,
        seed=args.seed,
    )
    stats, solution = solver.solve()
    print(json.dumps(stats.to_dict(), indent=2))

    if solution is not None:
        sol_list = solution.tolist()
        if args.verify:
            verified = is_valid_solution(sol_list)
            print(f"verified={verified}")
        if args.save_solution:
            Path(args.save_solution).write_text(json.dumps(sol_list))
        if args.show_board:
            print(format_board(sol_list))


def run_benchmark(args: argparse.Namespace) -> None:
    """CLI handler for benchmarking across multiple n values."""
    rows: List[dict] = []

    for n in args.n_values:
        solver = MinConflictsSolver(
            n=n,
            max_steps=args.max_steps,
            max_restarts=args.max_restarts,
            seed=args.seed,
        )
        stats, solution = solver.solve()

        # Verify solution if requested
        if args.verify and solution is not None:
            ok = is_valid_solution(solution.tolist())
            if not ok:
                raise RuntimeError(f"verification failed for n={n}")

        rows.append(stats.to_dict())
        print(
            f"n={n} solved={stats.solved} steps={stats.steps} time={stats.duration_sec:.3f}s"
        )

    # Save results to file if requested
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))


def format_board(positions: Sequence[int]) -> str:
    """
    Create ASCII representation of the board.

    Only works for small boards (n <= 30) to avoid huge output.
    Queens shown as 'Q', empty squares as '.'.
    """
    n = len(positions)
    if n > 30:
        return f"[board too large to display: n={n}]"

    lines = []
    for row in range(n):
        line = []
        for col in range(n):
            line.append("Q" if positions[col] == row else ".")
        lines.append(" ".join(line))
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description="Min-conflicts n-queens solver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Solve subcommand
    solve_parser = subparsers.add_parser("solve", help="solve a single instance")
    solve_parser.add_argument("--n", type=int, required=True, help="board size")
    solve_parser.add_argument(
        "--max-steps", type=int, default=None, help="max iterations per restart"
    )
    solve_parser.add_argument(
        "--max-restarts", type=int, default=3, help="number of restarts"
    )
    solve_parser.add_argument("--seed", type=int, default=None, help="random seed")
    solve_parser.add_argument("--verify", action="store_true", help="validate solution")
    solve_parser.add_argument(
        "--show-board", action="store_true", help="print ASCII board"
    )
    solve_parser.add_argument(
        "--save-solution", type=str, default=None, help="save to JSON file"
    )
    solve_parser.set_defaults(func=run_solve)

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="run solver across many n")
    bench_parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[10, 100, 1000, 10000, 100000, 1000000],
        help="list of n values to test",
    )
    bench_parser.add_argument(
        "--max-steps", type=int, default=None, help="max iterations per restart"
    )
    bench_parser.add_argument(
        "--max-restarts", type=int, default=3, help="number of restarts"
    )
    bench_parser.add_argument("--seed", type=int, default=None, help="random seed")
    bench_parser.add_argument(
        "--verify", action="store_true", help="validate each solution"
    )
    bench_parser.add_argument(
        "--output", type=str, default=None, help="save results to JSON file"
    )
    bench_parser.set_defaults(func=run_benchmark)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
