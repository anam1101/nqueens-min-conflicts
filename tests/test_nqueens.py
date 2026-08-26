"""Tests for the min-conflicts n-queens solver."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nqueens import MinConflictsSolver, is_valid_solution, format_board


@pytest.mark.parametrize("n", [1, 4, 8, 20, 100])
def test_solver_finds_valid_solution(n):
    """The solver should find a conflict-free board for solvable sizes."""
    solver = MinConflictsSolver(n=n, seed=42)
    stats, solution = solver.solve()

    assert stats.solved, f"solver failed for n={n}"
    assert solution is not None
    assert len(solution) == n
    assert is_valid_solution(solution.tolist())


@pytest.mark.parametrize("n", [2, 3])
def test_unsolvable_sizes_rejected(n):
    """n=2 and n=3 have no solutions and should be rejected up front."""
    with pytest.raises(ValueError, match="no solutions"):
        MinConflictsSolver(n=n)


def test_invalid_size_rejected():
    with pytest.raises(ValueError, match="n must be >= 1"):
        MinConflictsSolver(n=0)


def test_seed_is_reproducible():
    """Same seed must produce the same board."""
    a = MinConflictsSolver(n=50, seed=7).solve()[1]
    b = MinConflictsSolver(n=50, seed=7).solve()[1]
    assert a.tolist() == b.tolist()


def test_validator_rejects_shared_row():
    # Two queens on row 0 -> invalid.
    assert not is_valid_solution([0, 0, 2, 3])


def test_validator_rejects_shared_diagonal():
    # (0,0) and (1,1) sit on the same down-right diagonal.
    assert not is_valid_solution([0, 1, 3, 2])


def test_validator_accepts_known_solution():
    # Standard 8-queens solution.
    assert is_valid_solution([0, 4, 7, 5, 2, 6, 1, 3])


def test_validator_rejects_out_of_range_row():
    assert not is_valid_solution([0, 9, 2, 3])


def test_stats_are_populated():
    stats, _ = MinConflictsSolver(n=30, seed=1).solve()
    assert stats.n == 30
    assert stats.steps >= 0
    assert stats.duration_sec > 0
    assert "duration_sec" in stats.to_dict()


def test_board_rendering():
    board = format_board([0, 4, 7, 5, 2, 6, 1, 3])
    assert board.count("Q") == 8
    assert format_board(list(range(40))).startswith("[board too large")
