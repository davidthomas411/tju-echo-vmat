from pathlib import Path

from .base_adapter import BaseAdapter, CaseInfo


class ExampleAdapter(BaseAdapter):
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def prepare_case(self, case_id: str) -> CaseInfo:
        data_root = self.data_root
        if (data_root / case_id / case_id).is_dir():
            data_root = data_root / case_id
        if not (data_root / case_id).is_dir():
            raise FileNotFoundError(f"Case '{case_id}' not found under {data_root}")
        return CaseInfo(case_id=case_id, data_dir=data_root, source="example")
