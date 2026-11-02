from __future__ import annotations

from typing import Protocol


class SourceError(Exception):
    """The data source could not answer (network down, unknown series...)."""


class DataSource(Protocol):
    name: str

    def get_series(self, indicator: str, country: str) -> dict[int, float]:
        """Year -> value for one indicator and ISO3 country. Missing years omitted."""
        ...

    def source_url(self, indicator: str, country: str) -> str:
        """A human-readable page where the user can check the number."""
        ...

    def extra_countries(self) -> dict[str, tuple[str, tuple[str, ...]]]:
        """Countries (code -> (name, aliases)) this source adds to the catalog."""
        ...
