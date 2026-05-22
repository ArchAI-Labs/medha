__all__ = ["app"]


def __getattr__(name: str):  # lazy import so _noop_embedder is importable before _app.py exists
    if name == "app":
        from medha.cli._app import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
