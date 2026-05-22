from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from medha.config import Settings
from medha.core import Medha
from medha.exceptions import ConfigurationError
from medha.interfaces.embedder import BaseEmbedder

app = typer.Typer(help="Medha cache management CLI.")

_LOTUS = r"""
                                             .                                                      
                                            ...                                                     
                                            ...                                                     
                                           .....                                                    
                                           ++.+-.                                                   
                                        ..-..-..-..                                                 
                                        +.#+.#-+#++.                                                
                                     ++.+-.+-.+-.+-.+..                                             
                                ................... ..+.                                            
                                ..++.++.++.++.++.++.++.++#..                                        
                                .. .. .....................                                         
                            ...++++++++++-+.....++-++-++.++....                                     
                         ....... ............ . ... .............  .                                
                         ...+-.--.++.-+.... ..  ....+-.+-.-+..+....                                 
                       ..............               ..................                              
                    . +#.+..+.#+..+....     .#      ..##..#..#..#.#+..                              
  ..               ...+#.++.##.++......   .... ..        .+..#+-#+--.-.                             
  ..               ............... ..      ......       .................                           
  ...++   ...-+.-+--+--+-++-...    ..   ...+-.+-.-..       .....-.--.--.--.-+.#.   ..+...           
  ................... .... ...         ...... ..... .       .....  ............  ........  .        
 . .----------+-+--+--+-+..       ..-#+-++-+--++--+-++.         ..+.+-+.-.--.-+.-+.-+.-....         
    ........... ........ ...    ..............  ... ........       . ........   ........ ..         
    -+..+......   .-#.-..      ...++.++.+++.   ..+-.+-.+-...      ....-...  ......-...-.            
    ......       ........    ...........  .  .    ............     .........   . .......            
   .+.++.        .++-..     ..++-+++++-#.         .#+-+-++-#+..     .++.....     ..-+.-.            
    .+-.+.   -+.-+.-+ .   ..+-.+-..  .     -.-#+....    .-+..+.-.   ..+..+..+..  .+.-+..            
    ......   ..........   .... .....       .... ....   ..........   ...........  .......            
  ...+..-... .-........   ..++..      .-++.++#++.++...     ..++..   .-.....--..  ...-+--.           
  ........  ............  ........ ......................  .......  .....+.....  .+......           
  .+.++.     --+..   .... .++-.    .#+-++.-  ...-+.++.+.    .+.-.  +.-    .-+..    ..+...           
 ........    ... .     .   ....   ....  ......       ....    .  .  ...    ... .    ......           
 ...+.+..    -#...     .....+-.  .--.-..     .      ...-+.. .--......     ...-.    ...-.. .         
 ......    .......     .. .....   .....             ...... .....   ..     .....    . ......         
 ....+#   ..#-++            ..+ . +-.                 .+.. .+...          .-+.+...   +-.            
 ..#-+#..   .--...                 ..                       ..            ..--.    .+-.#...         
 .........   .....                 ..                .... .               .....  . .......          
 . ..+.-+.   .....       .. ..                         +.. ..           ...-... ....+-..            
    ......  ......        .....          ...              .                 . ...........           
      +--#-+-. ...#..    ...+++..+   ..+..-..    ..+..++-+...   #...   ...   .-..#.-....            
     ..  ...... .....    .........   .......... .............. ...     ...   ...........            
             ...........    ...##+..+.......++..#-....... ......--...............                   
        .....  . .. . ............          ..... .....   ..........................                
         .+-# .#..#..+-... ..+#. ...        ...-#.... . .#..++.-+..#..+.-..-...+.+.                 
          .+#..    ..+..#...-.#...      ..#  .  .+.....        .  .... ....-..                      
          .................+........................... ....#.. ................                    
                            .. ...-..  .......................                                      
                               ....... ....-.....-.....-....                                        
                                     .     ----..                                                   
                                         . ......      ..                                           
                                         ...  .-..                                                  
                                           ..    . .                                                
                                         .+.-. .+...                                                
                                           ..+......                                                
                                           ..-......                                                
                                            .....                                                   
                                            ..                                                                                                                      

                                         - MEDHA -
"""


@app.command()
def logo() -> None:
    """Print the Medha lotus logo."""
    typer.echo(_LOTUS)


def _resolve_embedder(settings: Settings) -> BaseEmbedder:
    et = settings.embedder_type
    if et == "_noop":
        from medha.cli._noop_embedder import _NoOpEmbedder
        return _NoOpEmbedder()
    if et == "fastembed":
        from medha.embeddings.fastembed_adapter import FastEmbedAdapter
        return FastEmbedAdapter(model_name=settings.fastembed_model)
    if et == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY env var is not set.")
        from medha.embeddings.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key=api_key)
    if et == "cohere":
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("COHERE_API_KEY env var is not set.")
        from medha.embeddings.cohere_adapter import CohereAdapter
        return CohereAdapter(api_key=api_key)
    if et == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY env var is not set.")
        from medha.embeddings.gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=api_key)
    raise ConfigurationError(f"Unknown embedder_type: '{et}'")


async def _build_medha(collection: str, settings: Settings) -> Medha:
    """Instantiate and start a Medha instance for the given collection."""
    embedder = _resolve_embedder(settings)
    m = Medha(
        collection_name=collection,
        embedder=embedder,
        settings=settings,
    )
    await m.start()
    return m


@app.command()
def stats(
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
) -> None:
    """Show structural stats for a collection.

    Reports collection name, backend type, and entry counts (main + templates).
    Does NOT report hit rate or latency: those are in-process, non-persistent
    metrics unavailable from a fresh CLI invocation.
    """
    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            main_count = await m._backend.count(m._collection_name)
            tmpl_count = await m._backend.count(m._template_collection)
            backend_name = type(m._backend).__name__
        typer.echo(f"Collection : {coll}")
        typer.echo(f"Backend    : {backend_name} ({settings.backend_type})")
        typer.echo(f"Entries    : {main_count} (main)  {tmpl_count} (templates)")

    asyncio.run(_run())


@app.command()
def warm(
    file: Path = typer.Argument(..., help="Path to a .json or .jsonl file of cache entries."),
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
    ttl: Optional[int] = typer.Option(
        None, "--ttl", help="TTL in seconds for each stored entry."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Number of entries per upsert batch."
    ),
) -> None:
    """Warm the cache from a JSON or JSONL file.

    Requires a real embedder: set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and
    install the matching extra (pip install 'medha-archai[fastembed]').
    """
    settings = Settings()
    coll = collection or settings.collection

    if settings.embedder_type == "_noop":
        typer.echo(
            "Error: 'warm' requires a real embedder.\n"
            "Set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and install the extra:\n"
            "  pip install 'medha-archai[fastembed]'",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        stored = 0

        def _on_progress(done: int, total: int) -> None:
            typer.echo(f"Progress: {done}/{total} entries stored.")

        async with m:
            try:
                stored = await m.warm_from_file(
                    str(file),
                    batch_size=batch_size,
                    on_progress=_on_progress,
                )
            except RuntimeError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=1)

        typer.echo(f"Warmed {stored} entries into '{coll}'.")

    asyncio.run(_run())


@app.command()
def invalidate(
    question: str = typer.Argument(..., help="Question text to remove from the cache."),
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
) -> None:
    """Remove a cached entry by question text."""
    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            removed = await m.invalidate(question)
        typer.echo("Removed." if removed else "Not found.")

    asyncio.run(_run())


@app.command("invalidate-collection")
def invalidate_collection(
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Confirm destructive deletion of the entire collection."
    ),
) -> None:
    """Delete all entries in a collection (destructive).

    Pass --yes to confirm. Without --yes this command exits without deleting anything.
    """
    settings = Settings()
    coll = collection or settings.collection

    if not yes:
        typer.echo(
            f"Warning: this will delete all entries in '{coll}'. "
            "Pass --yes to confirm.",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            deleted = await m.invalidate_collection()
        typer.echo(f"Deleted {deleted} entries from '{coll}'.")

    asyncio.run(_run())


@app.command()
def expire(
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
) -> None:
    """Delete all expired entries from a collection."""
    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            deleted = await m.expire()
        typer.echo(f"Deleted {deleted} expired entries.")

    asyncio.run(_run())


@app.command()
def dedup(
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
) -> None:
    """Remove duplicate entries that share the same generated query hash.

    Requires pandas: pip install pandas.
    """
    try:
        import pandas  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: 'dedup' requires pandas.\n"
            "Install it with: pip install pandas",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            removed = await m.dedup_collection()
        typer.echo(f"Removed {removed} duplicate entries.")

    asyncio.run(_run())


@app.command()
def export(
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Output file path. Defaults to stdout."
    ),
    format_: str = typer.Option(
        "csv", "--format", help="Output format: csv or json."
    ),
) -> None:
    """Export all cache entries to CSV or JSON.

    Requires pandas: pip install pandas.
    Writes to --output if given, otherwise prints to stdout.
    """
    try:
        import pandas  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: 'export' requires pandas.\n"
            "Install it with: pip install pandas",
            err=True,
        )
        raise typer.Exit(code=1)

    if format_ not in ("csv", "json"):
        typer.echo(f"Error: unsupported format '{format_}'. Choose csv or json.", err=True)
        raise typer.Exit(code=1)

    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            try:
                df = await m.export_to_dataframe()
            except ConfigurationError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=1)

        if format_ == "csv":
            content = df.to_csv(index=False)
        else:
            content = df.to_json(orient="records", indent=2)

        if output is None:
            typer.echo(content)
        else:
            output.write_text(content, encoding="utf-8")
            typer.echo(f"Exported {len(df)} entries to '{output}'.")

    asyncio.run(_run())


@app.command()
def feedback(
    question: str = typer.Argument(..., help="Question text to record feedback for."),
    collection: Optional[str] = typer.Option(
        None, "--collection", help="Collection name (env: MEDHA_COLLECTION)."
    ),
    correct: bool = typer.Option(
        False, "--correct/--no-correct", help="Mark the cached answer as correct."
    ),
    incorrect: bool = typer.Option(
        False, "--incorrect/--no-incorrect", help="Mark the cached answer as incorrect."
    ),
) -> None:
    """Record correctness feedback for a cached entry.

    Pass --correct or --incorrect (mutually exclusive).
    Works with the default _NoOpEmbedder; no real embedder is needed.
    """
    if correct and incorrect:
        typer.echo("Error: --correct and --incorrect are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    if not correct and not incorrect:
        typer.echo("Error: pass --correct or --incorrect.", err=True)
        raise typer.Exit(code=1)

    settings = Settings()
    coll = collection or settings.collection

    async def _run() -> None:
        try:
            m = await _build_medha(coll, settings)
        except (ConfigurationError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        async with m:
            found = await m.feedback(question, correct=correct)
        typer.echo("Feedback recorded." if found else "Entry not found.")

    asyncio.run(_run())
