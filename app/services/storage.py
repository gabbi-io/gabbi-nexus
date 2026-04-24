from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from app.config import CASES_DIR
from app.models import CaseRecord, SourceChunk


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonCaseRepository:
    def __init__(self) -> None:
        CASES_DIR.mkdir(parents=True, exist_ok=True)

    def _case_dir(self, case_id: str) -> Path:
        return CASES_DIR / case_id

    def _case_file(self, case_id: str) -> Path:
        return self._case_dir(case_id) / 'case.json'

    def _chunks_file(self, case_id: str) -> Path:
        return self._case_dir(case_id) / 'chunks.json'

    def _uploads_dir(self, case_id: str) -> Path:
        return self._case_dir(case_id) / 'uploads'

    def create_case(self, title: str) -> CaseRecord:
        case_id = uuid4().hex[:10]
        case_dir = self._case_dir(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        self._uploads_dir(case_id).mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        record = CaseRecord(case_id=case_id, title=title, created_at=now, updated_at=now)
        self.save_case(record)
        self.save_chunks(case_id, [])
        return record

    def save_case(self, record: CaseRecord) -> None:
        case_dir = self._case_dir(record.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        record.updated_at = utc_now_iso()
        self._case_file(record.case_id).write_text(record.model_dump_json(indent=2), encoding='utf-8')

    def get_case(self, case_id: str) -> CaseRecord:
        path = self._case_file(case_id)
        if not path.exists():
            raise FileNotFoundError(f'Case {case_id} not found')
        return CaseRecord.model_validate_json(path.read_text(encoding='utf-8'))

    def list_cases(self) -> List[CaseRecord]:
        records: List[CaseRecord] = []
        for case_file in CASES_DIR.glob('*/case.json'):
            records.append(CaseRecord.model_validate_json(case_file.read_text(encoding='utf-8')))
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def save_uploaded_file(self, case_id: str, filename: str, content: bytes) -> Path:
        upload_dir = self._uploads_dir(case_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid4().hex[:8]}_{filename}"
        path = upload_dir / safe_name
        path.write_bytes(content)
        return path

    def save_chunks(self, case_id: str, chunks: Iterable[SourceChunk]) -> None:
        serialized = [chunk.model_dump() for chunk in chunks]
        self._chunks_file(case_id).write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding='utf-8')

    def load_chunks(self, case_id: str) -> List[SourceChunk]:
        path = self._chunks_file(case_id)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding='utf-8'))
        return [SourceChunk.model_validate(item) for item in raw]
