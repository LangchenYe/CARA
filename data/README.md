# CloudCons inputs

Run `python download_data.py` from the code directory to download the eight original Arrow objects into `data/raw`.

| Collection | Sources | Use |
| --- | --- | --- |
| `cloudcons` | Azure, Borg-d, Borg-e, Huawei | Complete decision cohorts for placement quality |
| `cloudcons_hourly` | Azure, Borg-d, Borg-e, Huawei | All hourly CPU series for query execution |

The decision cohorts contain 150, 127, 68 and 575 workloads. The hourly sources contain 10,800, 414, 618 and 174 series. Each query study uses every series over the common observed interval, with all valid eight-hour windows. The hourly reader applies deterministic linear interpolation to the ten missing Huawei CPU observations in that interval.

```sh
python download_data.py --collection cloudcons
python download_data.py --collection cloudcons_hourly
```

The complete download is approximately 128 MB. The original Arrow files are read directly; preprocessing is performed in memory. The data repository is [CloudCons-ds](https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds). Source attribution and applicable licenses are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
