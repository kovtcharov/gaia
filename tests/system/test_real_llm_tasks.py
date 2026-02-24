# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Tier 4: Real-LLM Task Validation

These tests use a live LLM (Lemonade or Claude) and validate that the agent
produces *correct* code — not just structurally valid files, but code whose
output matches known-good results.

Run with:
    python -m pytest tests/system/test_real_llm_tasks.py --real-llm -v

All tests use the real_agent_workspace fixture (no ChatSDK mock) and are
guarded by the require_lemonade fixture (auto-skipped without --real-llm).

Test catalogue:
  1.  Sorting algorithm correctness  (merge sort)
  2.  OOP class with property methods (Temperature converter)
  3.  String utilities               (palindrome, reverse_words, count_vowels)
  4.  Data structures                (Stack class)
  5.  Prime number sieve             (Sieve of Eratosthenes)
  6.  Fix broken code                (buggy binary search)
  7.  Multi-file Python package      (geometry/)
  8.  Recursive algorithm            (Tower of Hanoi)
  9.  CLI tool executed via subprocess (factorial calculator)
  10. Multi-turn memory              (remember Python version, use in next query)
"""

import ast
import importlib.util
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_module(path: Path, name: str) -> types.ModuleType:
    """Import a .py file from an absolute path and return the module object."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Insert workspace dir so intra-package imports work
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _run(args, cwd, timeout=30):
    """Run a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ---------------------------------------------------------------------------
# 1. Sorting algorithm correctness
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_merge_sort_correctness(real_agent_workspace):
    """
    Agent creates mergesort.py with merge_sort(lst).
    Verify it correctly sorts integers, empty list, single-element, reversed.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create mergesort.py in the directory {ws} with a single function "
        "merge_sort(lst) that returns a new sorted list using the merge sort algorithm. "
        "Do not modify the input list. The function must work for any "
        "list of comparable elements."
    )

    f = ws / "mergesort.py"
    assert f.exists(), "mergesort.py must be created"
    ast.parse(f.read_text())  # valid Python

    mod = _load_module(f, "mergesort")
    assert hasattr(mod, "merge_sort"), "merge_sort function must exist"

    assert mod.merge_sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
    assert mod.merge_sort([]) == []
    assert mod.merge_sort([42]) == [42]
    assert mod.merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert mod.merge_sort([1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# 2. OOP class with property methods
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_temperature_converter_class(real_agent_workspace):
    """
    Agent creates temperature.py with a Temperature class.
    Verify Celsius-to-Fahrenheit and Celsius-to-Kelvin conversions are correct.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create temperature.py in {ws} with a Temperature class. "
        "The constructor takes a single float argument: the temperature in Celsius. "
        "Add two methods: to_fahrenheit() which returns the Fahrenheit equivalent "
        "(formula: C * 9/5 + 32), and to_kelvin() which returns the Kelvin equivalent "
        "(formula: C + 273.15). Both methods return floats."
    )

    f = ws / "temperature.py"
    assert f.exists(), "temperature.py must be created"
    ast.parse(f.read_text())

    mod = _load_module(f, "temperature")
    assert hasattr(mod, "Temperature"), "Temperature class must exist"
    T = mod.Temperature

    assert T(0).to_fahrenheit() == pytest.approx(32.0)
    assert T(100).to_fahrenheit() == pytest.approx(212.0)
    assert T(-40).to_fahrenheit() == pytest.approx(-40.0)  # intersection point
    assert T(0).to_kelvin() == pytest.approx(273.15)
    assert T(100).to_kelvin() == pytest.approx(373.15)
    assert T(-273.15).to_kelvin() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. String utility functions
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_string_utilities(real_agent_workspace):
    """
    Agent creates string_utils.py with three functions.
    Verify edge cases including empty strings, mixed case, and spaces.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create string_utils.py in {ws} with three standalone functions:\n"
        "1. is_palindrome(s: str) -> bool  — returns True if s reads the same "
        "forwards and backwards (case-insensitive, ignore spaces).\n"
        "2. reverse_words(s: str) -> str  — reverses the ORDER of words in the "
        "string (words are space-separated), preserving each word's characters.\n"
        "3. count_vowels(s: str) -> int  — returns the count of vowel characters "
        "(a, e, i, o, u, case-insensitive) in s."
    )

    f = ws / "string_utils.py"
    assert f.exists(), "string_utils.py must be created"
    ast.parse(f.read_text())

    mod = _load_module(f, "string_utils")

    # is_palindrome
    assert mod.is_palindrome("racecar") is True
    assert mod.is_palindrome("A man a plan a canal Panama") is True
    assert mod.is_palindrome("hello") is False
    assert mod.is_palindrome("") is True
    assert mod.is_palindrome("a") is True

    # reverse_words
    assert mod.reverse_words("hello world") == "world hello"
    assert mod.reverse_words("one two three") == "three two one"
    assert mod.reverse_words("solo") == "solo"

    # count_vowels
    assert mod.count_vowels("hello") == 2
    assert mod.count_vowels("AEIOU") == 5
    assert mod.count_vowels("bcdfg") == 0
    assert mod.count_vowels("") == 0


# ---------------------------------------------------------------------------
# 4. Data structure: Stack class
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_stack_data_structure(real_agent_workspace):
    """
    Agent creates stack.py with a Stack class backed by a Python list.
    Verify push, pop, peek, is_empty, and size behaviour.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create stack.py in {ws} with a Stack class that uses a Python list internally. "
        "Implement: push(item) — adds item to top, "
        "pop() — removes and returns top item (raises IndexError if empty), "
        "peek() — returns top item without removing (raises IndexError if empty), "
        "is_empty() — returns True when stack has no items, "
        "size() — returns number of items currently in the stack."
    )

    f = ws / "stack.py"
    assert f.exists(), "stack.py must be created"
    ast.parse(f.read_text())

    mod = _load_module(f, "stack")
    assert hasattr(mod, "Stack"), "Stack class must exist"
    S = mod.Stack

    s = S()
    assert s.is_empty() is True
    assert s.size() == 0

    s.push(1)
    s.push(2)
    s.push(3)
    assert s.size() == 3
    assert s.is_empty() is False
    assert s.peek() == 3
    assert s.size() == 3  # peek doesn't remove

    assert s.pop() == 3
    assert s.pop() == 2
    assert s.size() == 1
    assert s.pop() == 1
    assert s.is_empty() is True

    with pytest.raises((IndexError, Exception)):
        s.pop()  # popping empty stack must raise


# ---------------------------------------------------------------------------
# 5. Prime number sieve
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_sieve_of_eratosthenes(real_agent_workspace):
    """
    Agent creates primes.py with sieve_of_eratosthenes(n).
    Verify it returns the correct list of primes up to n.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create primes.py in {ws} with a single function sieve_of_eratosthenes(n: int) "
        "that returns a sorted list of all prime numbers <= n. "
        "Use the Sieve of Eratosthenes algorithm. "
        "sieve_of_eratosthenes(0) and sieve_of_eratosthenes(1) return []."
    )

    f = ws / "primes.py"
    assert f.exists(), "primes.py must be created"
    ast.parse(f.read_text())

    mod = _load_module(f, "primes")
    assert hasattr(mod, "sieve_of_eratosthenes"), "sieve_of_eratosthenes must exist"
    sieve = mod.sieve_of_eratosthenes

    assert sieve(0) == []
    assert sieve(1) == []
    assert sieve(2) == [2]
    assert sieve(10) == [2, 3, 5, 7]
    assert sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert sieve(50) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


# ---------------------------------------------------------------------------
# 6. Fix broken code (real quality gate recovery)
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_fix_broken_binary_search(real_agent_workspace):
    """
    A deliberately broken binary_search.py is written to the workspace.
    The agent is asked to fix it.  Verify the fixed version is correct.
    """
    agent, ws = real_agent_workspace

    broken = textwrap.dedent("""\
        def binary_search(lst, target):
            left, right = 0, len(lst) - 1
            while left <= right:
                mid = (left + right) // 2
                if lst[mid] == target:
                    return mid
                elif lst[mid] < target:
                    right = mid - 1   # BUG: should be left = mid + 1
                else:
                    left = mid + 1    # BUG: should be right = mid - 1
            return -1
    """)
    broken_file = ws / "binary_search.py"
    broken_file.write_text(broken)

    agent.process_query(
        f"The file {broken_file} contains a binary search implementation "
        "with incorrect update logic — the left/right pointers are moved in "
        "the wrong directions. Fix the function so that "
        "binary_search([1,2,3,4,5], 3) returns 2 and "
        "binary_search([1,2,3,4,5], 10) returns -1."
    )

    assert broken_file.exists(), "binary_search.py must still exist"
    ast.parse(broken_file.read_text())  # must be valid Python

    mod = _load_module(broken_file, "binary_search_fixed")
    assert hasattr(mod, "binary_search"), "binary_search function must exist"
    bs = mod.binary_search

    assert bs([1, 2, 3, 4, 5], 3) == 2
    assert bs([1, 2, 3, 4, 5], 1) == 0
    assert bs([1, 2, 3, 4, 5], 5) == 4
    assert bs([1, 2, 3, 4, 5], 10) == -1
    assert bs([], 1) == -1


# ---------------------------------------------------------------------------
# 7. Multi-file Python package
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_geometry_package(real_agent_workspace):
    """
    Agent creates a geometry/ package with __init__.py and shapes.py.
    Verify Circle and Rectangle area/perimeter calculations.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create a Python package called geometry/ inside {ws}. "
        "It needs two files:\n"
        f"1. {ws}/geometry/__init__.py — imports Circle and Rectangle from geometry.shapes\n"
        f"2. {ws}/geometry/shapes.py — defines:\n"
        "   - Circle(radius): has area() returning π*r² and perimeter() returning 2*π*r\n"
        "   - Rectangle(width, height): has area() returning w*h and "
        "perimeter() returning 2*(w+h)\n"
        "Use math.pi for π. Both classes must work when imported as "
        "'from geometry import Circle, Rectangle'."
    )

    pkg_dir = ws / "geometry"
    assert pkg_dir.is_dir(), "geometry/ package directory must exist"
    assert (pkg_dir / "__init__.py").exists(), "geometry/__init__.py must exist"
    assert (pkg_dir / "shapes.py").exists(), "geometry/shapes.py must exist"

    # Verify valid Python
    ast.parse((pkg_dir / "__init__.py").read_text())
    ast.parse((pkg_dir / "shapes.py").read_text())

    # Import and test
    import math

    if str(ws) not in sys.path:
        sys.path.insert(0, str(ws))

    import importlib

    geo = importlib.import_module("geometry")
    importlib.reload(geo)

    Circle = geo.Circle
    Rectangle = geo.Rectangle

    c = Circle(5)
    assert c.area() == pytest.approx(math.pi * 25, rel=1e-6)
    assert c.perimeter() == pytest.approx(2 * math.pi * 5, rel=1e-6)

    r = Rectangle(3, 4)
    assert r.area() == pytest.approx(12)
    assert r.perimeter() == pytest.approx(14)

    r2 = Rectangle(7, 7)
    assert r2.area() == pytest.approx(49)


# ---------------------------------------------------------------------------
# 8. Recursive algorithm: Tower of Hanoi
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_tower_of_hanoi(real_agent_workspace):
    """
    Agent creates hanoi.py with hanoi(n, source, target, auxiliary).
    Verify move count is 2^n - 1 and all moves are valid peg-to-peg tuples.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create hanoi.py in {ws} with a function hanoi(n, source='A', target='C', auxiliary='B') "
        "that solves the Tower of Hanoi puzzle for n discs. "
        "The function must return a list of (from_peg, to_peg) tuples representing "
        "each move in order. For example, hanoi(1) returns [('A', 'C')]. "
        "hanoi(3) must return exactly 7 moves."
    )

    f = ws / "hanoi.py"
    assert f.exists(), "hanoi.py must be created"
    ast.parse(f.read_text())

    mod = _load_module(f, "hanoi")
    assert hasattr(mod, "hanoi"), "hanoi function must exist"
    h = mod.hanoi

    # 1 disc: 1 move
    moves1 = h(1)
    assert len(moves1) == 1, f"hanoi(1) should have 1 move, got {len(moves1)}"
    assert moves1[0] == ("A", "C"), f"hanoi(1) wrong move: {moves1[0]}"

    # 2 discs: 3 moves
    moves2 = h(2)
    assert len(moves2) == 3, f"hanoi(2) should have 3 moves, got {len(moves2)}"

    # 3 discs: 7 moves, all pegs in {A, B, C}
    moves3 = h(3)
    assert len(moves3) == 7, f"hanoi(3) should have 7 moves, got {len(moves3)}"
    valid_pegs = {"A", "B", "C"}
    for frm, to in moves3:
        assert frm in valid_pegs, f"Invalid source peg: {frm!r}"
        assert to in valid_pegs, f"Invalid target peg: {to!r}"
        assert frm != to, f"Move from peg to itself: {frm!r}"

    # 4 discs: 15 moves
    moves4 = h(4)
    assert len(moves4) == 15, f"hanoi(4) should have 15 moves, got {len(moves4)}"


# ---------------------------------------------------------------------------
# 9. CLI tool executed via subprocess
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_cli_factorial_tool(real_agent_workspace):
    """
    Agent creates cli_factorial.py that reads a non-negative integer from
    argv[1] and prints its factorial to stdout.
    Verify with known values by calling the script via subprocess.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        f"Create cli_factorial.py in {ws} — a command-line script that:\n"
        "1. Reads a single non-negative integer from sys.argv[1]\n"
        "2. Computes its factorial\n"
        "3. Prints just the integer result to stdout (nothing else)\n"
        "Example: `python cli_factorial.py 5` prints `120`"
    )

    f = ws / "cli_factorial.py"
    assert f.exists(), "cli_factorial.py must be created"
    ast.parse(f.read_text())

    def run_factorial(n):
        rc, out, err = _run([sys.executable, str(f), str(n)], cwd=ws)
        assert rc == 0, f"cli_factorial.py {n} failed (rc={rc}): {err}"
        return int(out)

    assert run_factorial(0) == 1
    assert run_factorial(1) == 1
    assert run_factorial(5) == 120
    assert run_factorial(10) == 3628800
    assert run_factorial(12) == 479001600


# ---------------------------------------------------------------------------
# 10. Multi-turn memory: agent remembers context across queries
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
def test_multi_turn_memory_retention(real_agent_workspace):
    """
    Query 1: tell the agent the project uses Python 3.12 and SQLite.
    Query 2: ask it to create config.py with project metadata.
    Verify config.py contains both "3.12" and "SQLite".
    """
    agent, ws = real_agent_workspace

    # First turn: establish facts
    agent.process_query(
        f"Working directory is {ws}. "
        "Remember these project facts for later: "
        "the project is called MyApp, it uses Python 3.12, "
        "and it stores data in SQLite. "
        "Acknowledge by saying 'facts stored'."
    )

    # Second turn: use those facts
    agent.process_query(
        f"Create config.py in {ws} with a Python dict or constants "
        "documenting the project metadata we discussed. "
        "The file must contain the project name, Python version, and database technology."
    )

    cfg = ws / "config.py"
    assert cfg.exists(), "config.py must be created"
    ast.parse(cfg.read_text())  # valid Python

    content = cfg.read_text()
    assert "3.12" in content, f"config.py must mention Python version 3.12\n{content}"
    assert "SQLite" in content or "sqlite" in content.lower(), (
        f"config.py must mention SQLite\n{content}"
    )
    assert "MyApp" in content or "myapp" in content.lower(), (
        f"config.py must mention project name\n{content}"
    )
