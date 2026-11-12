# Third-Party Notices

## CloudCons benchmark data

This artifact uses four processed decision-trace objects from **CloudCons: A
Comprehensive End-to-End Benchmark for Cloud Resource Consolidation** (KDD
2026 Datasets and Benchmarks Track), by Xiaobin Zhang, Lefei Shen, Mouxiang
Chen, Zhuo Li, Hongkai Li, Han Fu, Jianling Sun, Xiaoxue Ren, and Chenghao Liu.

- Dataset repository: <https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds>
- Code and provenance statement: <https://github.com/dmwyd/CloudCons>
- Paper: <https://arxiv.org/abs/2606.13513>

The upstream repository states that CloudCons derives from publicly released
Huawei Cloud, Microsoft Azure Public Dataset V2, and Google ClusterData 2019
traces and is distributed in compliance with the original CC BY 4.0 and MIT
licenses. The Arrow files remain third-party data; CARA's MIT license does not
relicense them. Users must retain attribution and comply with the applicable
upstream terms.

The downloader also retrieves the four CloudCons hourly forecasting Arrow objects
for Azure, Borg-d, Borg-e, and Huawei. They retain the same upstream attribution
and applicable source terms. The downloaded Arrow objects are unchanged. The
query-workload reader uses the CPU channel and deterministically interpolates
ten missing Huawei CPU values in memory; it does not redistribute a modified
dataset or claim forecasting accuracy from this processing workload.

## Dependencies

Python packages retain their upstream licenses. Dependencies are installed separately.

