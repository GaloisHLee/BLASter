#!/usr/bin/env python3
"""Compatibility entry point for running BLASter from the source tree."""

from blaster.app import main


if __name__ == "__main__":
    raise SystemExit(main())
