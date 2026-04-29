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

Create local configuration files from the committed examples:

```bash
copy .env.example .env
copy config\storage.example.yaml config\storage.yaml
copy config\providers.example.yaml config\providers.yaml
copy config\rules.example.yaml config\rules.yaml
```

Initialize local storage:

```bash
stock-picker storage init --config config/storage.yaml
```

Validate local storage:

```bash
stock-picker storage validate --config config/storage.yaml
```

Register schema files into the SQLite metadata catalog:

```bash
stock-picker storage register-schemas --config config/storage.yaml
```

Import a local CSV into the curated current store:

```bash
stock-picker storage import-curated-csv --config config/storage.yaml --dataset daily_prices --input data/manual_input/daily_prices.csv --as-of-date 2026-04-28
```

Manual CSV templates live under `examples/manual_input/`. Copy a template into `data/manual_input/`, replace the sample rows with real data, and import it.

The first CSV import path is intentionally manual. It validates against the registered schema catalog, fills standard lineage fields when missing, fills nullable missing columns as null values, writes Parquet under `data/curated/current/<dataset>/`, and records `data_batches` plus `curated_versions` metadata in SQLite.

Inspect a curated dataset after import:

```bash
stock-picker storage inspect-curated --config config/storage.yaml --dataset daily_prices
```

Run the MVP quality gate for curated current datasets:

```bash
stock-picker storage check-quality --config config/storage.yaml
```

To check a single dataset:

```bash
stock-picker storage check-quality --config config/storage.yaml --dataset daily_prices
```

Create a logical snapshot manifest from the current curated versions:

```bash
stock-picker storage create-snapshot --config config/storage.yaml --as-of-date 2026-04-28
```

Inspect a stored snapshot:

```bash
stock-picker storage inspect-snapshot --config config/storage.yaml --snapshot-id snapshot_20260428_001
```

## Local Configuration

Runtime configuration files are intentionally local-only:

- `.env`
- `config/storage.yaml`
- `config/providers.yaml`
- `config/rules.yaml`

Commit only example files such as `.env.example` and `config/*.example.yaml`. Put provider tokens in `.env`, machine-specific storage paths in `config/storage.yaml`, provider choices in `config/providers.yaml`, and rule experiments in `config/rules.yaml`.

## Portability

Do not hardcode local machine paths in code. Storage paths should come from `config/storage.yaml`, credentials should come from environment variables or ignored local files, and generated local directories such as `.venv/` and `data/` should not be committed. To recreate the project on a new machine, install the package, copy the example config files, set local secrets, run `stock-picker storage init`, and then run `stock-picker storage validate`.
