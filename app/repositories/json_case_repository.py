from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonCaseRepository:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _case_dir(self, case_id: str) -> Path:
        path = self.base_path / case_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _case_file(self, case_id: str) -> Path:
        return self._case_dir(case_id) / "case.json"

    def _files_dir(self, case_id: str) -> Path:
        path = self._case_dir(case_id) / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_case(self, case_id: str, data: dict[str, Any]) -> None:
        self._write_case(case_id, data)

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        path = self._case_file(case_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_cases(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.base_path.glob("*/case.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                items.append({
                    "id": payload.get("id"),
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                    "documents_count": len(payload.get("documents", [])),
                    "has_analysis": payload.get("analysis") is not None,
                    "has_diagnostic": payload.get("diagnostic") is not None,
                })
            except Exception:
                continue
        return items

    def save_uploaded_file(self, case_id: str, filename: str, content: bytes) -> str:
        file_path = self._files_dir(case_id) / filename
        file_path.write_bytes(content)
        return str(file_path)

    def add_document(self, case_id: str, document: dict[str, Any]) -> None:
        payload = self.get_case(case_id) or {}
        payload.setdefault("documents", []).append(document)
        self._write_case(case_id, payload)

    def update_case(self, case_id: str, patch: dict[str, Any]) -> None:
        payload = self.get_case(case_id)
        if not payload:
            raise ValueError(f"Case {case_id} not found")
        payload.update(patch)
        self._write_case(case_id, payload)

    def append_chat_history(self, case_id: str, item: dict[str, Any]) -> None:
        payload = self.get_case(case_id)
        if not payload:
            raise ValueError(f"Case {case_id} not found")
        payload.setdefault("chat_history", []).append(item)
        self._write_case(case_id, payload)

    def save_export(self, case_id: str, kind: str, content: str, suffix: str = "json") -> str:
        path = self._case_dir(case_id) / "exports"
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{kind}.{suffix}"
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def persist_case(self, case_id: str) -> None:
        return None

    def _write_case(self, case_id: str, data: dict[str, Any]) -> None:
        self._case_file(case_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
