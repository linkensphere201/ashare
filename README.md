# Stock Picker

Stock-first A-share daily quantitative research and report system.

This repository contains the implementation code. Design documents and decisions live in the workspace documentation project:

```text
../2026-04-20-a-share-stock-picker/
```

## Current MVP

The current implementation supports the first real-data research loop:

- Initialize local storage and metadata.
- Register machine-readable curated and metadata schemas.
- Fetch raw Tushare data into the raw store.
- Probe Tushare APIs and expected fields with small requests.
- Promote Tushare raw batches into standard curated Parquet datasets.
- Run curated data quality checks.
- Create snapshot manifests for reproducible strategy runs.
- Rank Strategy Candidate 001 v2 candidates.
- Run a minimal forward-return diagnostic backtest.

Reports, dashboard UI, portfolio sizing, broker execution, and richer backtest diagnostics are future scope.

## Technology Stack

- Python `>=3.11,<3.15`.
- Polars for local DataFrame processing.
- PyArrow and Parquet for curated datasets.
- DuckDB for analytical SQL over Parquet.
- SQLite for metadata catalog, snapshots, and run state.
- Tushare for first real-data provider integration.
- Pytest for tests.

## Environment Setup

Run these commands from the `stock-picker/` repository root.

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the venv Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m stock_picker.cli --help
```

Install the project and development dependencies:

```powershell
python -m pip install -e .[dev]
```

After installation, this console script should be available inside the activated environment:

```powershell
stock-picker --help
```

## Local Configuration

Runtime files are intentionally local-only and should not be committed:

- `.env`
- `config/storage.yaml`
- `config/providers.yaml`
- `config/rules.yaml`
- `data/`
- `.venv/`

Create local config files from the committed examples:

```powershell
copy config\storage.example.yaml config\storage.yaml
copy config\providers.example.yaml config\providers.yaml
copy config\rules.example.yaml config\rules.yaml
```

Credentials are read from environment variables. For Tushare, set `TUSHARE_TOKEN` in the active shell before running provider commands:

```powershell
$env:TUSHARE_TOKEN = "your-token-here"
```

Do not print tokens in logs or commit them. `.env.example` is ignored in this workspace because it may contain local secret placeholders during development.

## First-Time Initialization

Initialize configured storage directories and the SQLite metadata catalog:

```powershell
stock-picker storage init --config config/storage.yaml
```

Validate configured storage paths and metadata:

```powershell
stock-picker storage validate --config config/storage.yaml
```

Register schema YAML files into the metadata catalog:

```powershell
stock-picker storage register-schemas --config config/storage.yaml
```

## Real Tushare Data Flow

The normal MVP path is:

```text
Tushare API -> raw CSV batches -> curated Parquet -> quality check -> snapshot -> strategy ranking/backtest
```

Supported first-pass Tushare datasets:

| Dataset | Tushare API | Curated Target | Purpose |
| --- | --- | --- | --- |
| `security_master` | `stock_basic` | `security_master` | Symbol identity and active universe |
| `trading_calendar` | `trade_cal` | `trading_calendar` | Trading-day alignment |
| `daily_prices` | `daily` | `daily_prices` | Prices, MACD, liquidity, forward returns |
| `moneyflow_dc` | `moneyflow_dc` | `capital_flow_or_chip` | Main-fund flow strength |
| `cyq_perf` | `cyq_perf` | `capital_flow_or_chip` | Winner rate / profit-chip condition |

Probe a small provider request before running larger fetches:

```powershell
stock-picker provider probe --source tushare --api moneyflow_dc --ts-code 600519.SH --trade-date 20260428
stock-picker provider probe --source tushare --api cyq_perf --ts-code 600519.SH --trade-date 20260428
```

Fetch raw provider data:

```powershell
stock-picker provider fetch --config config/storage.yaml --source tushare --dataset security_master --as-of-date 2026-04-28
stock-picker provider fetch --config config/storage.yaml --source tushare --dataset trading_calendar --start-date 2025-04-28 --end-date 2026-04-28 --as-of-date 2026-04-28
stock-picker provider fetch --config config/storage.yaml --source tushare --dataset daily_prices --start-date 2025-04-28 --end-date 2026-04-28 --as-of-date 2026-04-28
stock-picker provider fetch --config config/storage.yaml --source tushare --dataset moneyflow_dc --start-date 2025-04-28 --end-date 2026-04-28 --as-of-date 2026-04-28
```

`cyq_perf` should be fetched by `--ts-code` first because the API is stock-code oriented:

```powershell
stock-picker provider fetch --config config/storage.yaml --source tushare --dataset cyq_perf --ts-code 600519.SH --start-date 2025-04-28 --end-date 2026-04-28 --as-of-date 2026-04-28
```

Promote raw batches into curated current Parquet:

```powershell
stock-picker storage promote-raw --config config/storage.yaml --source tushare --dataset security_master --as-of-date 2026-04-28
stock-picker storage promote-raw --config config/storage.yaml --source tushare --dataset trading_calendar --as-of-date 2026-04-28
stock-picker storage promote-raw --config config/storage.yaml --source tushare --dataset daily_prices --as-of-date 2026-04-28
stock-picker storage promote-raw --config config/storage.yaml --source tushare --dataset moneyflow_dc --as-of-date 2026-04-28
stock-picker storage promote-raw --config config/storage.yaml --source tushare --dataset cyq_perf --as-of-date 2026-04-28
```

Run quality checks and create a snapshot:

```powershell
stock-picker storage check-quality --config config/storage.yaml
stock-picker storage create-snapshot --config config/storage.yaml --as-of-date 2026-04-28
stock-picker storage inspect-snapshot --config config/storage.yaml --snapshot-id snapshot_20260428_001
```

## Strategy Commands

Strategy Candidate 001 v2 uses:

- `moneyflow_dc.net_amount_rate > 0`
- MACD golden cross from `daily_prices.close`, EMA 12 / 26 / 9
- `cyq_perf.winner_rate > 80`
- Top 10 candidate output
- Ranking by `net_amount_rate desc`, then `winner_rate desc`, then `20d_return desc`
- 20-trading-day forward-return diagnostic backtest

Rank latest candidates from a snapshot:

```powershell
stock-picker strategy rank-candidate-001 --config config/storage.yaml --snapshot-id snapshot_20260428_001 --top 10
```

Run the diagnostic backtest:

```powershell
stock-picker strategy backtest-candidate-001 --config config/storage.yaml --snapshot-id snapshot_20260428_001 --holding-days 20 --top 10
```

The output is a research/watchlist result, not direct buy or sell advice.

## CLI Reference

### Storage

| Command | Purpose |
| --- | --- |
| `stock-picker storage init` | Create configured storage directories and SQLite metadata tables |
| `stock-picker storage validate` | Validate configured storage directories and metadata |
| `stock-picker storage register-schemas` | Register schema YAML files into SQLite catalog tables |
| `stock-picker storage import-curated-csv` | Import a local CSV into curated current Parquet |
| `stock-picker storage promote-raw` | Promote a raw provider batch into curated current Parquet |
| `stock-picker storage inspect-curated` | Inspect current curated metadata and Parquet columns for one dataset |
| `stock-picker storage check-quality` | Run MVP data quality checks against curated datasets |
| `stock-picker storage create-snapshot` | Create a logical snapshot manifest from current curated versions |
| `stock-picker storage inspect-snapshot` | Inspect a stored snapshot manifest |

Useful examples:

```powershell
stock-picker storage inspect-curated --config config/storage.yaml --dataset daily_prices
stock-picker storage check-quality --config config/storage.yaml --dataset daily_prices
```

### Provider

| Command | Purpose |
| --- | --- |
| `stock-picker provider probe` | Run a small Tushare API request and check expected fields |
| `stock-picker provider fetch` | Fetch a Tushare dataset into raw CSV storage and record metadata |

Supported probe APIs:

```text
stock_basic, trade_cal, daily, moneyflow_dc, moneyflow_ths, cyq_perf
```

Supported fetch datasets:

```text
security_master, trading_calendar, daily_prices, moneyflow_dc, cyq_perf
```

Provider commands read `TUSHARE_TOKEN` from the environment and never print the token.

### Strategy

| Command | Purpose |
| --- | --- |
| `stock-picker strategy rank-candidate-001` | Rank Strategy Candidate 001 v2 candidates from a snapshot |
| `stock-picker strategy backtest-candidate-001` | Run forward-return diagnostics for Strategy Candidate 001 v2 |

## Manual CSV Debug Path

Manual CSV templates live under `examples/manual_input/`. They are kept for development, tests, and emergency debugging only.

The normal workflow should fetch raw data from Tushare instead of requiring user-supplied CSV files. If needed, CSV import still works:

```powershell
stock-picker storage import-curated-csv --config config/storage.yaml --dataset daily_prices --input data/manual_input/daily_prices.csv --as-of-date 2026-04-28
```

The import validates against the registered schema catalog, fills standard lineage fields, writes Parquet under `data/curated/current/<dataset>/`, and records `data_batches` plus `curated_versions` metadata in SQLite.

## Testing

Run tests with a workspace-local pytest temp/cache directory on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .tmp\pytest -o cache_dir=.tmp\pytest-cache
```

The local temp/cache flags avoid permission problems with the global Windows temp directory.

## Portability

Do not hardcode local machine paths in code. Storage paths should come from `config/storage.yaml`, credentials should come from environment variables or ignored local files, and generated local directories such as `.venv/` and `data/` should not be committed.

To recreate the project on a new machine:

1. Install Python.
2. Create and activate `.venv`.
3. Run `python -m pip install -e .[dev]`.
4. Copy example config files into local runtime config files.
5. Set `TUSHARE_TOKEN` in the local environment.
6. Run `stock-picker storage init`.
7. Run `stock-picker storage validate`.
8. Run `stock-picker storage register-schemas`.
