# Manual Input CSV Templates

These templates define the smallest manual CSV shape for the MVP curated import loop.

Copy the files you want to use into `data/manual_input/`, replace the sample rows with real data, then run `stock-picker storage import-curated-csv`.

The importer fills standard lineage fields when they are missing:

- `source`
- `source_batch_id`
- `data_version`
- `created_at`

For date-based datasets, the importer also fills physical partition fields when missing:

- `trade_year` and `trade_month` from `trade_date`
- `event_year` and `event_month` from `event_date`

Nullable schema fields may be omitted from manual CSV files. They will be written as null values in the curated Parquet output.

## Import Examples

```powershell
Copy-Item examples\manual_input\security_master.csv data\manual_input\security_master.csv
.\.venv\Scripts\stock-picker storage import-curated-csv --dataset security_master --input data/manual_input/security_master.csv --as-of-date 2026-04-28
```

```powershell
Copy-Item examples\manual_input\daily_prices.csv data\manual_input\daily_prices.csv
.\.venv\Scripts\stock-picker storage import-curated-csv --dataset daily_prices --input data/manual_input/daily_prices.csv --as-of-date 2026-04-28
```

Run `storage register-schemas` before the first import.
