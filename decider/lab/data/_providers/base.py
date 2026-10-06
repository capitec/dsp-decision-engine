from __future__ import annotations

from abc import abstractmethod

import polars as pl
from pydantic import ConfigDict, PrivateAttr

from decider.lab.data._keys import check_key, suggest_key
from decider.lab.data._load import LoadedData, _fingerprint
from decider.exceptions import DeciderError
from decider.registry import BaseRegistryModule


class DatasetError(DeciderError, ValueError):
    """A dataset can't be found, inferred or configured."""

    _STATUS_CODE = 400


class DatasetProvider(BaseRegistryModule, root=True, group="datasets"):
    """One dataset: where it lives and how to read it; subclass it for a new source.

    `key` is the data's unique record key; `load()` rejects duplicate or null keys.

    Example::

        class Warehouse(DatasetProvider):
            type: t.Literal["warehouse"] = "warehouse"
            table: str

            def scan(self):
                return pl.scan_pyarrow_dataset(open_table(self.table))
    """

    model_config = ConfigDict(extra="forbid")
    key: tuple[str, ...] | None = None
    _name: str = PrivateAttr(default="")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def scan(self) -> pl.LazyFrame:
        """The dataset as a lazy frame."""

    def schema(self) -> dict[str, str]:
        """Column name to dtype, without reading the data.

        Example::

            provider.schema()   # {"client_id": "Int64"}
        """
        return {name: str(dtype) for name, dtype in self.scan().collect_schema().items()}

    def head(self, n: int = 50) -> pl.DataFrame:
        """The first `n` rows.

        Example::

            provider.head(5)
        """
        return self.scan().head(n).collect()

    def fingerprint(self, frame: pl.DataFrame) -> str | None:
        """The content fingerprint a run records; override it when the source has a cheaper identity."""
        return _fingerprint(frame)

    def load(self) -> LoadedData:
        """The whole dataset as `LoadedData`, its key checked.

        Example::

            data = provider.load()
            data.frame, data.suggested_key
        """
        frame = self.scan().collect()
        if self.key:
            check_key(frame, self.key)
        return LoadedData.from_frame(frame, dataset=self.name, suggested_key=self.key or suggest_key(frame.columns),
                                     fingerprint=self.fingerprint(frame))


DatasetProvider.lazy({"file": "decider.lab.data._providers.files", "folder": "decider.lab.data._providers.files"})
