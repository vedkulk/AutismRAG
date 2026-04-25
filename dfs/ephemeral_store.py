from __future__ import annotations

"""
EphemeralDFSStore — session-scoped encrypted+fragmented storage for the
uploaded patient PDF.

Lifecycle:
    1. RAGPipeline creates one store at startup. Root is a system tempdir
       (e.g. /var/folders/.../rag_dfs_XXXX/) — never under the project tree.
    2. On upload, store(bytes):
         - zstd compress
         - AES-256-GCM encrypt with a fresh per-file key
         - split ciphertext into N fragments
         - Shamir-split the AES key (3-of-5)
         - scatter fragments + key shares across node_* subdirs
       Returns a StoredHandle with on-disk paths + nonce.
    3. reconstruct(handle): read fragments, recombine key from threshold
       shares, decrypt, decompress, return plaintext bytes.
    4. delete_active() wipes the current handle. wipe() tears the whole
       tempdir down. Both are also driven by atexit and a weakref finalizer
       so a normal interpreter exit cleans up after itself.

Threat model is "best effort against casual disk inspection mid-session"
plus "nothing patient-readable persists after exit". Shamir on a single
host is mostly defence-in-depth — the AES-GCM is doing the real work.
SSD wear-levelling means overwrite-then-unlink is symbolic, not a guarantee.
"""

import atexit
import logging
import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .compression_manager import CompressionManager
from .encryption import EncryptionManager
from .erasure_manager import merge_fragments, split_into_fragments
from .key_manager import KeyManager

logger = logging.getLogger(__name__)

TEMPDIR_PREFIX = "rag_dfs_"


@dataclass
class StoredHandle:
    fragment_paths: List[Path]
    key_share_paths: List[Path]
    nonce: bytes
    threshold: int
    plaintext_size: int = 0
    metadata: dict = field(default_factory=dict)


def sweep_orphans() -> int:
    """Delete tempdirs left over from prior crashed sessions.

    Called once at app startup. A clinician who lost a session just re-uploads.
    """
    base = Path(tempfile.gettempdir())
    if not base.exists():
        return 0

    swept = 0
    try:
        for entry in base.iterdir():
            if entry.is_dir() and entry.name.startswith(TEMPDIR_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
                swept += 1
    except OSError as e:
        logger.warning("sweep_orphans: scan of %s failed: %s", base, e)

    if swept:
        logger.info("Swept %d orphan DFS tempdir(s)", swept)
    return swept


class EphemeralDFSStore:
    def __init__(
        self,
        num_fragments: int = 10,
        num_key_shares: int = 5,
        key_threshold: int = 3,
        num_nodes: int = 5,
    ) -> None:
        if key_threshold > num_key_shares:
            raise ValueError("key_threshold cannot exceed num_key_shares")
        if num_fragments < 1 or num_nodes < 1:
            raise ValueError("num_fragments and num_nodes must be >= 1")

        self._num_fragments = num_fragments
        self._num_key_shares = num_key_shares
        self._key_threshold = key_threshold
        self._num_nodes = num_nodes

        self._cm = CompressionManager()
        self._em = EncryptionManager()
        self._km = KeyManager()

        self._root = Path(tempfile.mkdtemp(prefix=TEMPDIR_PREFIX))
        self._handle: Optional[StoredHandle] = None

        # Best-effort cleanup on GC and on interpreter exit
        self._finalizer = weakref.finalize(self, _wipe_dir, self._root)
        _register_for_atexit(self)

        logger.info(
            "EphemeralDFSStore initialised at %s (fragments=%d, shares=%d, threshold=%d)",
            self._root, num_fragments, num_key_shares, key_threshold,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def has_active(self) -> bool:
        return self._handle is not None

    # ── store / reconstruct ─────────────────────────────────────────────

    def store(self, data: bytes) -> StoredHandle:
        """Compress → encrypt → fragment → scatter. Replaces any active handle."""
        if self._handle is not None:
            self.delete_active()

        compressed = self._cm.compress(data)
        key = self._em.generate_key()
        ciphertext, nonce = self._em.encrypt(compressed, key)
        fragments = split_into_fragments(ciphertext, self._num_fragments)
        shares = self._km.split_key(
            key,
            num_shares=self._num_key_shares,
            threshold=self._key_threshold,
        )

        frag_paths: List[Path] = []
        for i, frag in enumerate(fragments):
            d = self._node_dir(i % self._num_nodes)
            p = d / f"frag_{i:03d}.bin"
            p.write_bytes(frag)
            frag_paths.append(p)

        share_paths: List[Path] = []
        for i, share in enumerate(shares):
            d = self._node_dir(i % self._num_nodes)
            p = d / f"share_{i:03d}.txt"
            p.write_text(share)
            share_paths.append(p)

        # Drop the in-memory key reference
        del key

        self._handle = StoredHandle(
            fragment_paths=frag_paths,
            key_share_paths=share_paths,
            nonce=nonce,
            threshold=self._key_threshold,
            plaintext_size=len(data),
        )
        logger.info(
            "DFS stored %d-byte payload as %d fragments across %d nodes",
            len(data), len(frag_paths), self._num_nodes,
        )
        return self._handle

    def reconstruct(self, handle: Optional[StoredHandle] = None) -> bytes:
        h = handle or self._handle
        if h is None:
            raise RuntimeError("EphemeralDFSStore: no active handle to reconstruct")

        ciphertext = merge_fragments([p.read_bytes() for p in h.fragment_paths])
        shares = [p.read_text().strip() for p in h.key_share_paths[: h.threshold]]
        key = self._km.reconstruct_key(shares)
        try:
            compressed = self._em.decrypt(ciphertext, h.nonce, key)
        finally:
            del key
        plaintext = self._cm.decompress(compressed)
        return plaintext

    # ── wipe ────────────────────────────────────────────────────────────

    def delete_active(self) -> None:
        """Securely unlink the active handle's on-disk artifacts."""
        if self._handle is None:
            return

        for p in (*self._handle.fragment_paths, *self._handle.key_share_paths):
            _secure_unlink(p)

        # Remove now-empty node dirs
        for entry in list(self._root.iterdir()):
            if entry.is_dir():
                try:
                    entry.rmdir()
                except OSError:
                    pass  # not empty — leave for full wipe()

        self._handle = None
        logger.info("DFS active handle wiped")

    def wipe(self) -> None:
        """Tear down the entire tempdir. The store is unusable after this."""
        self.delete_active()
        _wipe_dir(self._root)
        try:
            self._finalizer.detach()
        except Exception:
            pass

    # ── helpers ─────────────────────────────────────────────────────────

    def _node_dir(self, i: int) -> Path:
        d = self._root / f"node_{i}"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ══════════════════════════════════════════════════════════════════════════
#  module-level cleanup helpers
# ══════════════════════════════════════════════════════════════════════════

def _secure_unlink(path: Path) -> None:
    """Best-effort overwrite-then-delete. Symbolic on SSDs (wear-levelling)."""
    try:
        try:
            size = path.stat().st_size
            if size > 0:
                with path.open("r+b") as f:
                    f.write(b"\x00" * size)
                    f.flush()
                    os.fsync(f.fileno())
        except OSError:
            pass
        path.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("secure_unlink %s failed: %s", path, e)


def _wipe_dir(root: Path) -> None:
    try:
        shutil.rmtree(root, ignore_errors=True)
    except Exception as e:
        logger.warning("Failed to wipe DFS dir %s: %s", root, e)


_active_stores: List["weakref.ref[EphemeralDFSStore]"] = []


def _register_for_atexit(store: EphemeralDFSStore) -> None:
    _active_stores.append(weakref.ref(store))


def _atexit_cleanup() -> None:
    for ref in _active_stores:
        s = ref()
        if s is not None:
            try:
                s.wipe()
            except Exception:
                pass


atexit.register(_atexit_cleanup)
