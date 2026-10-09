"""Shared rich console for the command line output.

Writes to stderr so stdout stays free for what scripts read: paths, JSON and
copy commands.
"""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=True)
