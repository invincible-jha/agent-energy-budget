"""Deterministic prompt hasher using SHA-256.

The hash is used as the primary cache key for exact-match lookups before
any embedding similarity search is attempted.
"""
from __future__ import annotations

import hashlib


class PromptHasher:
    """SHA-256 based prompt hasher for exact cache key lookup.

    The hash is computed over the UTF-8 encoded prompt string and returned
    as a 64-character lowercase hex digest.  Identical prompts always
    produce the same hash; different prompts produce different hashes with
    extremely high probability.

    Example
    -------
    >>> hasher = PromptHasher()
    >>> hasher.hash("hello world")
    'b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576f2ff28af41b5d...'
    """

    def hash(self, prompt: str) -> str:
        """Compute the SHA-256 hex digest of *prompt*.

        Parameters
        ----------
        prompt:
            The raw prompt string to hash.

        Returns
        -------
        str
            64-character lowercase hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
