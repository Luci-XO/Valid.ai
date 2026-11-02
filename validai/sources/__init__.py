from .base import DataSource, SourceError
from .snapshot import SnapshotSource
from .worldbank import WorldBankSource

__all__ = ["DataSource", "SourceError", "SnapshotSource", "WorldBankSource", "from_spec"]


def from_spec(spec: str) -> DataSource:
    """Build a source from a string: ``worldbank`` or ``snapshot:path.json``."""
    if spec == "worldbank":
        return WorldBankSource()
    if spec.startswith("snapshot:"):
        return SnapshotSource(spec.split(":", 1)[1])
    raise ValueError(f"unknown source {spec!r} (use 'worldbank' or 'snapshot:<path>')")
