from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def write_report(path: Path, payload: dict):
    payload["runtime"] = {"python": sys.version.split()[0], "platform": platform.platform()}
    path.write_text(json.dumps(payload, indent=2, cls=Encoder), encoding="utf-8")
