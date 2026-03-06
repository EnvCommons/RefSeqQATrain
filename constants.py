import os
from pathlib import Path

if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data") / "data"
else:
    DATA_PATH = Path(__file__).parent / "data"

REFSEQTRAIN_JSONL = DATA_PATH / "refseqtrain.jsonl"
