"""Pure-Python cosine similarity computation.

Uses only the Python standard library (math module) — no numpy or
third-party dependencies.  Suitable for small-to-medium embedding
vectors where avoiding a heavy dependency is preferable.
"""
from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two embedding vectors.

    Cosine similarity is the dot product of *a* and *b* divided by the
    product of their L2 norms.  The result lies in [-1.0, 1.0] for
    arbitrary real-valued vectors, but is always in [0.0, 1.0] for
    non-negative embeddings produced by most language models.

    Parameters
    ----------
    a:
        First embedding vector.  Must have the same length as *b*.
    b:
        Second embedding vector.  Must have the same length as *a*.

    Returns
    -------
    float
        Cosine similarity in the range [-1.0, 1.0].
        Returns 0.0 when either vector is a zero vector.

    Raises
    ------
    ValueError
        If *a* and *b* have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: len(a)={len(a)}, len(b)={len(b)}. "
            "Both vectors must have the same number of dimensions."
        )

    dot_product: float = sum(x * y for x, y in zip(a, b))
    norm_a: float = math.sqrt(sum(x * x for x in a))
    norm_b: float = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)
