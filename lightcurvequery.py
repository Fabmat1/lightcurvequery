#!/usr/bin/env python
"""
Thin wrapper kept for backward compatibility.
Now simply delegates to the real package entry-point.
"""
from src.lightcurvequery_core.cli import main

if __name__ == "__main__":
    main()