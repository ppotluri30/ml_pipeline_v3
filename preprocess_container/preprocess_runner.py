"""Legacy wrapper retained for backward compatibility.

The authoritative implementation now resides in `main.py`.
This file allows existing container CMD or scripts referencing
`preprocess_runner.py` to continue to function.
"""
from __future__ import annotations

from main import main

if __name__ == "__main__":  # pragma: no cover
    main()