def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
