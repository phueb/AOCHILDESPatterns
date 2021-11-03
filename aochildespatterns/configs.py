from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    summaries = root / 'summaries'
