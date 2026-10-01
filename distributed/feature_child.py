"""Child process for the unchanged V2 25-query feature partition."""
import argparse
import json
from pathlib import Path

from worker import configure


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=int, required=True)
    p.add_argument('--part', type=int, required=True)
    p.add_argument('--parts', type=int, required=True)
    a = p.parse_args()
    _, pair = configure()
    records = json.loads(a.records.read_text(encoding='utf-8'))
    pair.run(records, a.source, a.output, a.part, a.parts)
