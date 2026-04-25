"""
Distributed File System (DFS) package.

Two surfaces:
  - EphemeralDFSStore (ephemeral_store.py): used by the RAG app to store the
    uploaded patient PDF as encrypted+fragmented blobs in a system tempdir
    for the lifetime of one session, then wipe.
  - The original CLI scripts (main.py, retrieve.py, transfer_*.py) are kept
    intact for the cross-device transfer flow and run as standalone scripts.
"""

from .ephemeral_store import EphemeralDFSStore, sweep_orphans

__all__ = ["EphemeralDFSStore", "sweep_orphans"]
