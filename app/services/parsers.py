from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from pptx import Presentation
from pypdf import PdfReader


class ParserService:
    def parse_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md", ".markdown"]:
            return self._parse_text(path)

        if suffix == ".csv":
            return self._parse_csv(path)

        if suffix in [".xlsx", ".xlsm", ".xls"]:
            return self._parse_xlsx(path)

        if suffix == ".pdf":
            return self._parse_pdf(path)

        if suffix == ".docx":
            return self._parse_docx(path)

        if suffix == ".pptx":
            return self._parse_pptx(path)

        return {
            "text": "",
            "tables": [],
            "metadata": {
                "warning": f"Formato não suportado: {suffix}",
                "filename": path.name,
                "suffix": suffix,
            },
        }

    def _parse_text(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "text": text,
            "tables": [],
            "metadata": {
                "pages": 1,
                "format": path.suffix.lower().replace(".", "") or "text",
                "filename": path.name,
                "characters": len(text),
            },
        }

    def _parse_txt(self, path: Path) -> dict[str, Any]:
        # Mantido por compatibilidade com chamadas antigas.
        return self._parse_text(path)

    def _parse_csv(self, path: Path) -> dict[str, Any]:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return {
            "text": self._df_to_text(df, path.name),
            "tables": [
                {
                    "sheet": "csv",
                    "columns": df.columns.tolist(),
                    "rows_preview": df.head(20).fillna("").to_dict(orient="records"),
                }
            ],
            "metadata": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "filename": path.name,
                "characters": int(len(self._df_to_text(df, path.name))),
            },
        }

    def _parse_xlsx(self, path: Path) -> dict[str, Any]:
        xl = pd.ExcelFile(path)
        texts = []
        tables = []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=str).fillna("")
            text = self._df_to_text(df, f"{path.name}:{sheet}")
            texts.append(text)
            tables.append(
                {
                    "sheet": sheet,
                    "columns": df.columns.tolist(),
                    "rows_preview": df.head(20).to_dict(orient="records"),
                }
            )

        full_text = "\n\n".join(texts)
        return {
            "text": full_text,
            "tables": tables,
            "metadata": {
                "sheets": xl.sheet_names,
                "filename": path.name,
                "characters": len(full_text),
            },
        }

    def _parse_pdf(self, path: Path) -> dict[str, Any]:
        reader = PdfReader(str(path))
        pages = []

        for i, page in enumerate(reader.pages, start=1):
            try:
                pages.append(f"[Página {i}]\n{page.extract_text() or ''}")
            except Exception:
                pages.append(f"[Página {i}]\n")

        text = "\n\n".join(pages)
        return {
            "text": text,
            "tables": [],
            "metadata": {
                "pages": len(reader.pages),
                "filename": path.name,
                "characters": len(text),
            },
        }

    def _parse_docx(self, path: Path) -> dict[str, Any]:
        doc = Document(str(path))
        texts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(texts)

        return {
            "text": text,
            "tables": [],
            "metadata": {
                "paragraphs": len(texts),
                "filename": path.name,
                "characters": len(text),
            },
        }

    def _parse_pptx(self, path: Path) -> dict[str, Any]:
        prs = Presentation(str(path))
        slides = []

        for i, slide in enumerate(prs.slides, start=1):
            texts = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(shape.text)

            slides.append(f"[Slide {i}]\n" + "\n".join(texts))

        text = "\n\n".join(slides)
        return {
            "text": text,
            "tables": [],
            "metadata": {
                "slides": len(prs.slides),
                "filename": path.name,
                "characters": len(text),
            },
        }

    def _df_to_text(self, df: pd.DataFrame, source_name: str) -> str:
        columns = ", ".join(map(str, df.columns.tolist()))
        preview = df.head(50).astype(str).to_csv(index=False)
        return f"[Tabela: {source_name}]\nColunas: {columns}\nPrévia:\n{preview}"
