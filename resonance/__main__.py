"""Lets you run the tool as `python -m resonance ...` (handy when the Scripts folder is not on PATH)."""
import sys

from .cli import main

sys.exit(main())
