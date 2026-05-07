from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from pptx import Presentation
from pypdf import PdfReader


class ParserService:
    """
    Parser com foco em não "sumir" com o conteúdo anexado.

    Correções:
    - .md/.markdown lidos como texto.
    - CSV tenta múltiplos separadores/encodings.
    - CSV/XLSX geram texto com mais linhas, não somente 50 fixas.
    - Erros de parsing voltam como metadata.warning sem quebrar o upload.
    """

    def __init__(self):
        self.max_table_text_rows = int(os.getenv("GABBI_PARSER_MAX_TABLE_TEXT_ROWS", "5000"))
        self.max_cell_chars = int(os.getenv("GABBI_PARSER_MAX_CELL_CHARS", "500"))

    def parse_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
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
        except Exception as exc:
            return {
                "text": "",
                "tables": [],
                "metadata": {
                    "warning": f"Falha ao processar arquivo: {exc}",
                    "filename": path.name,
                    "suffix": suffix,
                    "error": str(exc),
                },
            }

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
        return self._parse_text(path)

    def _parse_csv(self, path: Path) -> dict[str, Any]:
        df = self._read_csv_robust(path)
        df = self._clean_df(df)
        text = self._df_to_text(df, path.name)
        return {
            "text": text,
            "tables": [
                {
                    "sheet": "csv",
                    "columns": df.columns.tolist(),
                    "rows_preview": df.head(50).to_dict(orient="records"),
                    "rows": int(df.shape[0]),
                }
            ],
            "metadata": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "filename": path.name,
                "characters": int(len(text)),
            },
        }

    def _read_csv_robust(self, path: Path) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
        seps = [None, ";", ",", "\t", "|"]
        last_error = None
        for enc in encodings:
            for sep in seps:
                try:
                    if sep is None:
                        return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python", encoding=enc)
                    return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, encoding=enc)
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"Não foi possível ler CSV: {last_error}")

    def _parse_xlsx(self, path: Path) -> dict[str, Any]:
        xl = pd.ExcelFile(path)
        texts = []
        tables = []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=str).fillna("")
            df = self._clean_df(df)
            text = self._df_to_text(df, f"{path.name}:{sheet}")
            texts.append(text)
            tables.append(
                {
                    "sheet": sheet,
                    "columns": df.columns.tolist(),
                    "rows_preview": df.head(50).to_dict(orient="records"),
                    "rows": int(df.shape[0]),
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
            "metadata": {"pages": len(reader.pages), "filename": path.name, "characters": len(text)},
        }

    def _parse_docx(self, path: Path) -> dict[str, Any]:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        table_texts = []
        for t_idx, table in enumerate(doc.tables, start=1):
            rows = []
            for row in table.rows:
                rows.append(" | ".join(cell.text.strip() for cell in row.cells))
            if rows:
                table_texts.append(f"[Tabela DOCX {t_idx}]\n" + "\n".join(rows))
        text = "\n".join(paragraphs + table_texts)
        return {
            "text": text,
            "tables": [],
            "metadata": {"paragraphs": len(paragraphs), "docx_tables": len(doc.tables), "filename": path.name, "characters": len(text)},
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
            "metadata": {"slides": len(prs.slides), "filename": path.name, "characters": len(text)},
        }

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna("").astype(str)
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            df[col] = df[col].map(lambda v: str(v).strip()[: self.max_cell_chars])
        return df

    def _df_to_text(self, df: pd.DataFrame, source_name: str) -> str:
        columns = ", ".join(map(str, df.columns.tolist()))
        row_count = int(df.shape[0])
        rows_to_text = min(row_count, self.max_table_text_rows)
        preview = df.head(rows_to_text).astype(str).to_csv(index=False)
        suffix = ""
        if row_count > rows_to_text:
            suffix = f"\n[Texto tabular truncado para indexação: exibindo {rows_to_text} de {row_count} linhas. Consultas tabulares usam o arquivo completo via pandas.]"
        return (
            f"[Tabela: {source_name}]\n"
            f"Linhas: {row_count}\n"
            f"Colunas: {columns}\n"
            f"Dados:\n{preview}{suffix}"
        )
