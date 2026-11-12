"""Download the eight unmodified CloudCons objects used by the experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def restore(row, root, repository):
    target = root / row['path']
    if target.exists() and target.stat().st_size == row['bytes']:
        return f"Verified {row['path']}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.part')
    url = f"https://huggingface.co/datasets/{repository}/resolve/main/{row['upstream_path']}"
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'CARA-reproduction'})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as output:
            for block in iter(lambda: source.read(1 << 20), b''):
                output.write(block)
        if temporary.stat().st_size != row['bytes']:
            raise ValueError(f"The downloaded input differs from the experiment input: {row['path']}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return f"Downloaded {row['path']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--collection', choices=('all', 'cloudcons', 'cloudcons_hourly'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    manifest = json.loads((ROOT / 'data' / 'inputs.json').read_text())
    rows = [row for row in manifest['files'] if args.collection == 'all' or row['path'].startswith(args.collection + '/')]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(restore, row, args.root, manifest['repository']) for row in rows]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
