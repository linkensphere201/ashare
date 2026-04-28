# Stock Picker

Stock-first A-share daily quantitative research and report system.

This repository contains the implementation code. Design documents and decisions live in the workspace documentation project:

```text
../2026-04-20-a-share-stock-picker/
```

## MVP Scope

The first implementation focuses on the storage and schema foundation:

- Config-driven local storage paths.
- Machine-readable dataset and metadata schemas.
- SQLite metadata catalog initialization.
- Storage directory initialization and validation commands.

Strategy logic, data-provider adapters, reports, backtests, and the web dashboard will be added after this foundation is stable.

## Technology Stack

- Python for implementation.
- Polars for local DataFrame processing.
- DuckDB for analytical SQL over Parquet.
- SQLite for metadata catalog, reports, configs, and run state.
- Parquet for large curated, factor, signal, and backtest datasets.

## Quick Start

Create a virtual environment and install the project:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Initialize local storage:

```bash
stock-picker storage init --config config/storage.yaml
```

Validate local storage:

```bash
stock-picker storage validate --config config/storage.yaml
```

## Portability

Do not hardcode local machine paths in code. Storage paths should come from `config/storage.yaml`, credentials should come from environment variables or ignored local files, and generated local directories such as `.venv/` and `data/` should not be committed.
