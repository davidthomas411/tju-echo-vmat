from __future__ import annotations

from pathlib import Path

from .base_adapter import BaseAdapter, CaseInfo


class EsapiAdapter(BaseAdapter):
    """Placeholder for future ESAPI integration (Eclipse)."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root

    def prepare_case(self, case_id: str) -> CaseInfo:
        raise NotImplementedError(
            "ESAPI adapter is a placeholder. Implement Eclipse case loading before use."
        )
