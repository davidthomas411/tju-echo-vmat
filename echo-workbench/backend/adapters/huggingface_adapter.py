import time
from pathlib import Path

from .base_adapter import BaseAdapter, CaseInfo


class HuggingFaceAdapter(BaseAdapter):
    def __init__(
        self,
        repo_id: str,
        cache_dir: Path,
        token: str | None = None,
        subdir: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.token = token
        self.subdir = subdir

    def _local_repo_dir(self) -> Path:
        safe_name = self.repo_id.replace("/", "__")
        return self.cache_dir / safe_name

    def _resolve_case_dir(self, repo_dir: Path, case_id: str) -> Path | None:
        if self.subdir:
            candidate = repo_dir / self.subdir / case_id
            if candidate.is_dir():
                return candidate
        candidate = repo_dir / case_id
        if candidate.is_dir():
            return candidate
        candidate = repo_dir / "data" / case_id
        if candidate.is_dir():
            return candidate
        return None

    def prepare_case(self, case_id: str) -> CaseInfo:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HuggingFaceAdapter. Install it in the backend env."
            ) from exc

        repo_dir = self._local_repo_dir()
        case_dir = self._resolve_case_dir(repo_dir, case_id)
        download_seconds = 0.0
        if case_dir is None:
            start = time.time()
            allow_patterns = None
            if self.subdir:
                allow_patterns = [f"{self.subdir}/{case_id}/**"]
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=repo_dir,
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                token=self.token,
            )
            download_seconds = time.time() - start
            case_dir = self._resolve_case_dir(repo_dir, case_id)
        if case_dir is None:
            raise FileNotFoundError(
                f"Case '{case_id}' not found after download in {repo_dir}"
            )
        return CaseInfo(
            case_id=case_id,
            data_dir=case_dir.parent,
            download_seconds=download_seconds,
            source=f"huggingface:{self.repo_id}",
        )
