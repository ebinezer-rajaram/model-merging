"""Legacy CLI shim for adapter merging.

Use `python main.py merge ...` or `python -m merging.cli` instead.
"""

from __future__ import annotations

from merging.cli import main


if __name__ == "__main__":
    main()
