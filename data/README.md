# Input data

Run `python download_data.py` from the repository root to download four CloudCons decision files and four hourly forecasting files into `data/raw`. The download is about 128 MB. The downloader uses the repository revision and file checks recorded in `inputs.json`.

The decision files are used for consolidation experiments; the hourly files are used for query measurements. `benchmark_hourly_queries.py` documents the in-memory interpolation of ten missing Huawei CPU entries.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for sources and terms. Use `python download_data.py --collection cloudcons` for decision data only, or `python download_data.py --collection cloudcons_hourly` for hourly data only.
