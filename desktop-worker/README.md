# Stock Picker Desktop Worker

Electron shell for the local Stock Picker console.

The desktop app does not implement research logic directly. It starts the Python
`stock_picker.cli` module and streams command output back into the UI.

## Development

From this directory:

```powershell
npm install
npm run dev
```

The app expects the Python project to be installed in the parent repository
environment. If `.venv` exists, it uses `.venv\Scripts\python.exe`; otherwise it
falls back to `python`.

## Runtime Config

Copy the worker config example in the parent repository:

```powershell
copy ..\config\app-worker.example.yaml ..\config\app-worker.yaml
```

Set the worker token through the environment or `.env`:

```text
STOCK_APP_WORKER_TOKEN=your-worker-token
```

## Unit Test Plan

The desktop console has two layers:

- Python runtime: daily bundle, market status, candidate pool, stock analysis, workflow state, worker claim/result handling, and worker-triggered daily bundle upload.
- Electron shell: tray/window UI, Python command launching, log streaming, and notification wiring.

Unit tests should cover the Python runtime with synthetic factor-run fixtures and mock worker tasks. They must not call the real Tushare provider or a real stock-app backend.

Required Python scenarios:

- Daily bundle, market status, and candidate pool success paths, required-section validation, prohibited wording, sensitive-field rejection, change labels, invalid `top`, missing factor run, and empty factor CSV.
- Stock analysis success, missing symbol, missing factor run, empty factor CSV, risk notes, and no local absolute paths or investment-advice wording in output.
- Worker `run-once` success for app-triggered `stock_analysis`, UTF-8 BOM mock JSON, unsupported request types, missing `factor_run_id`, and missing HTTP worker token.
- Worker `daily-check` success, already-uploaded skip behavior, shared mock result upload payload, and failed upload paths.
- Workflow status, pause, failed step state, dry-run preflight requiring confirmation, JSON events, and resume behavior that skips completed steps.
- CLI help smoke for `publish`, `analysis`, `workflow`, and `app-worker`.

Electron checks should stay lightweight in unit test runs:

```powershell
npm.cmd run check
```

This performs syntax checks for the Electron main and preload scripts without launching the GUI.

## Manual Smoke Flow

Run Python desktop-console tests from the parent repository:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_desktop_console.py --basetemp E:\projects\project-manager\.tmp-stock-tests\pytest-desktop -o cache_dir=E:\projects\project-manager\.tmp-stock-tests\pytest-cache-desktop
```

Run the full Python test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp E:\projects\project-manager\.tmp-stock-tests\pytest-full -o cache_dir=E:\projects\project-manager\.tmp-stock-tests\pytest-cache-full
```

Run Electron syntax checks from this directory:

```powershell
npm.cmd run check
```

Optional local smoke against an existing Candidate 002 factor run:

```powershell
.\.venv\Scripts\python.exe -m stock_picker.cli publish build-daily-bundle --config config/storage.yaml --factor-run-id factor_002_latest_20260506 --top 5
.\.venv\Scripts\python.exe -m stock_picker.cli app-worker daily-check --config config/storage.yaml --worker-config config/app-worker.yaml --factor-run-id factor_002_latest_20260506 --mock-upload
.\.venv\Scripts\python.exe -m stock_picker.cli analysis stock --config config/storage.yaml --factor-run-id factor_002_latest_20260506 --symbol 600519.SH
.\.venv\Scripts\python.exe -m stock_picker.cli workflow stock-analysis --config config/storage.yaml --workflow-id smoke_stock_analysis --factor-run-id factor_002_latest_20260506 --symbol 600519.SH --trade-date 2026-05-06 --json-events
```

The optional smoke reads local reports only. It should not run real provider syncs or upload to a real backend.

## Worker Product Flow

The APP backend exposes one shared worker result upload interface. It accepts both daily bundles and stock-analysis job results and performs lightweight validation before saving them.

The worker has two responsibilities:

- APP-triggered task queue: a user clicks stock analysis in APP, APP backend creates a `stock_analysis` job with `pending` status, the worker polls and claims it, runs `stock_app_stock_analysis_v001`, then uploads the job result through the shared result upload interface.
- Worker-triggered daily publishing: APP has no user-triggered daily-analysis flow. The worker periodically runs local daily analysis, builds `daily_publish_bundle_v001`, skips if the same bundle hash was already uploaded, and uploads it through the same shared result upload interface.

`report_update` should not be created by APP backend and should not be claimed from the queue. Daily reports are global outputs; APP displays the latest result date and shows an outdated/update-not-complete state when the latest uploaded date is stale. Per-user holding risk matching is applied by APP backend using the bundle's public risk match rules.
