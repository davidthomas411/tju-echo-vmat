from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class CaseInfo:
    case_id: str
    data_dir: Path
    download_seconds: float = 0.0
    source: str = "local"


class BaseAdapter(Protocol):
    def prepare_case(self, case_id: str) -> CaseInfo:
        ...
