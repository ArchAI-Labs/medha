from typing import Any

__all__ = ["app"]

_CLI_EXTRA_HINT = (
    "The 'medha' CLI requires the [cli] extra (typer, rich).\n"
    "Install it with:  pip install \"medha-archai[cli]\""
)


def __getattr__(name: str) -> Any:  # lazy import so _noop_embedder is importable before _app.py exists
    if name == "app":
        try:
            from medha.cli._app import app
        except ImportError as exc:
            # Reached via the `medha` console script when the extra is missing;
            # the bare "No module named 'typer'" gives the user nothing to act on.
            raise ImportError(f"{_CLI_EXTRA_HINT}\n\nOriginal error: {exc}") from exc
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
