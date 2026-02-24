import typing as t
import json
from functools import lru_cache

from omegaconf import OmegaConf
from pydantic import Field, model_validator

from .namespaced import NamespacedModule
from ped.param.sources.core import DictVersionedSource
from ped.param.param_source import ParameterSourceProvider
from ped.param.context import config_source_provider
import ped.param  # noqa: F401 – ensures the "ped.param" OmegaConf resolver is registered


@lru_cache(maxsize=128)
def load_external_config(config_path: str) -> dict:
    """Load external module configuration from a JSON file (cached by path)."""
    with open(config_path, "r") as f:
        return json.load(f)


class ExternalModule(NamespacedModule):
    """A :class:`NamespacedModule` whose definition is loaded from an external
    JSON file.

    This enables reuse: a module configuration can be saved once via
    :meth:`save` and referenced from many flows.  Each reference can supply
    its own ``parameters`` to fill in ``${ped.param:params,<key>}``
    interpolation placeholders that appear in the saved JSON.

    Loading flow
    ------------
    1. The JSON file at ``config_path`` is read (result is cached by path).
    2. A ``params`` parameter source backed by ``parameters`` is pushed onto
       the OmegaConf resolver context.
    3. OmegaConf resolves all ``${ped.param:params,<key>}`` references in the
       loaded config.
    4. Any fields passed explicitly (other than ``config_path`` /
       ``parameters``) override what was loaded from the file.

    Parameters
    ----------
    config_path:
        Path to the JSON file produced by :meth:`save`.
    parameters:
        Key-value pairs exposed to OmegaConf as the ``params`` source so that
        the loaded config can be parameterized at construction time.
    """

    type: t.Literal["external"] = "external"  # type: ignore[assignment]

    config_path: str
    parameters: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description=(
            "Values injected as the 'params' OmegaConf source when resolving "
            "the loaded config. Use ${ped.param:params,<key>} placeholders in "
            "the saved JSON to reference these."
        ),
    )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Save the reusable module definition to ``config_path``.

        ``config_path``, ``parameters``, and ``type`` are excluded so the
        saved file contains only the portable structural definition.
        """
        with open(self.config_path, "w") as f:
            json.dump(
                self.model_dump(
                    exclude_defaults=True,
                    exclude={"config_path", "parameters", "type"},
                ),
                f,
                indent=4,
            )

    # ------------------------------------------------------------------ #
    # Config loading / parameterization                                   #
    # ------------------------------------------------------------------ #

    @model_validator(mode="before")
    @classmethod
    def load_config(cls, values: dict) -> dict:
        """Load the external JSON and merge it with any explicitly passed values."""
        config_path = values.get("config_path")
        if not config_path:
            return values

        loaded_config: dict = dict(load_external_config(config_path))
        parameters: dict = values.get("parameters", {})

        # Resolve OmegaConf interpolations (${ped.param:params,<key>}) using
        # the provided parameters as the "params" source.
        with config_source_provider(
            ParameterSourceProvider(
                sources={"params": DictVersionedSource(values=parameters)},
                inputs={},
            )
        ):
            loaded_config = OmegaConf.to_object(OmegaConf.create(loaded_config))

        # Explicitly passed values take precedence over the loaded config.
        values = loaded_config | values

        # Always identify this model as ExternalModule regardless of what
        # type was recorded in the file.
        values["type"] = "external"
        return values