from __future__ import annotations

import re
from collections import Counter
from typing import Any


class AnalysisService:
    COMMON_ENTITY_HINTS = {
        "cliente": ["cliente", "beneficiário", "beneficiario", "usuário", "usuario"],
        "incidente": ["incidente", "ticket", "chamado", "ocorrência", "ocorrencia"],
        "processo": ["processo", "fluxo", "etapa", "atividade"],
        "contrato": ["contrato", "plano", "apólice", "apolice"],
        "data": ["data", "prazo", "período", "periodo"],
        "valor": ["valor", "custo", "preço", "preco", "quantia"],
    }

    def generate_initial_analysis(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        total_chars = 0
        total_tables = 0
        findings = []
        warnings = []
        entities = Counter()
        rules_detected = []
        exceptions_detected = []
        file_types = Counter()

        for doc in documents:
            filename = doc.get("filename", "arquivo")
            parsed = doc.get("parsed", {})
            text = parsed.get("text", "") or ""
            tables = parsed.get("tables", []) or []
            total_chars += len(text)
            total_tables += len(tables)
            suffix = filename.split(".")[-1].lower() if "." in filename else "desconhecido"
            file_types[suffix] += 1

            if text:
                findings.append(f"{filename}: {min(len(text), 12000)} caracteres processados")
            if tables:
                findings.append(f"{filename}: {len(tables)} tabela(s) detectada(s)")
            if len(text) < 100:
                warnings.append(f"{filename}: pouco conteúdo textual extraído")

            lowered = text.lower()
            for entity, hints in self.COMMON_ENTITY_HINTS.items():
                if any(h in lowered for h in hints):
                    entities[entity] += 1

            for line in self._interesting_lines(text):
                low = line.lower()
                if any(k in low for k in ["se ", "quando ", "deve ", "somente", "apenas", "obrigatório", "obrigatorio"]):
                    rules_detected.append(self._clip(line))
                if any(k in low for k in ["exceto", "exceção", "excecao", "erro", "falha", "inconsist", "pendente"]):
                    exceptions_detected.append(self._clip(line))

        possible_entities = [e for e, _ in entities.most_common(8)]
        readiness = self._readiness_score(total_tables, total_chars, len(rules_detected), len(exceptions_detected))
        priority_recommendations = self._build_priority_recommendations(total_tables, total_chars, possible_entities)

        summary = (
            f"Foram analisados {len(documents)} documento(s), com {total_chars} caracteres de conteúdo e {total_tables} tabela(s) detectada(s). "
            f"As principais entidades percebidas foram: {', '.join(possible_entities) if possible_entities else 'não identificadas com segurança'}."
        )

        return {
            "documents_count": len(documents),
            "file_types": dict(file_types),
            "total_text_chars": total_chars,
            "total_tables": total_tables,
            "summary": summary,
            "possible_entities": possible_entities,
            "rules_detected": rules_detected[:12],
            "exceptions_detected": exceptions_detected[:12],
            "findings": findings[:20],
            "warnings": warnings[:10],
            "automation_readiness_score": readiness,
            "priority_recommendations": priority_recommendations,
        }

    def format_answer(self, question: str, evidences: list[dict[str, Any]], analysis: dict[str, Any], mode: str = "executive") -> dict[str, Any]:
        evidence_files = list(dict.fromkeys([e.get("filename", "arquivo") for e in evidences]))
        summary = self._fallback_answer(question, evidences, analysis, mode)
        return {
            "summary": summary,
            "answer_text": summary,
            "evidences": evidences,
            "evidence_files": evidence_files,
        }

    def generate_structured_diagnostic(self, documents: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
        entities = analysis.get("possible_entities", [])
        rules = analysis.get("rules_detected", [])
        exceptions = analysis.get("exceptions_detected", [])
        suggestions = [item["name"] for item in analysis.get("priority_recommendations", [])]
        recommended_capabilities = self._recommended_capabilities(analysis)

        return {
            "objective_of_process": self._infer_objective(documents, entities),
            "inputs": [doc.get("filename") for doc in documents],
            "outputs": ["resumo analítico", "diagnóstico estruturado", "blueprint inicial", "workflow n8n inicial"],
            "main_entities": entities,
            "decision_rules": rules[:10],
            "exceptions": exceptions[:10],
            "bottlenecks": self._infer_bottlenecks(analysis),
            "risks": self._infer_risks(analysis),
            "automation_suggestions": suggestions,
            "recommended_capabilities": recommended_capabilities,
            "automation_readiness_score": analysis.get("automation_readiness_score", 0),
            "priority_level": self._priority_label(analysis.get("automation_readiness_score", 0)),
            "executive_summary": self._diagnostic_summary(analysis, suggestions, recommended_capabilities),
        }

    def _fallback_answer(self, question: str, evidences: list[dict[str, Any]], analysis: dict[str, Any], mode: str) -> str:
        ev_lines = []
        for ev in evidences[:4]:
            excerpt = (ev.get("excerpt") or "").strip().replace("\n", " ")[:260]
            ev_lines.append(f"- **{ev.get('filename', 'arquivo')}**: {excerpt}")
        if not ev_lines:
            ev_lines.append("- Não houve evidências suficientes recuperadas para responder com profundidade.")

        if mode == "technical":
            return (
                "## Resposta técnica\n\n"
                f"**Pergunta:** {question}\n\n"
                f"**Entidades percebidas:** {', '.join(analysis.get('possible_entities', [])) or 'nenhuma'}\n\n"
                "### Evidências recuperadas\n"
                + "\n".join(ev_lines)
            )

        return (
            "## Resumo executivo\n\n"
            f"Com base no material analisado, a pergunta **\"{question}\"** aponta para um cenário com entidades como **{', '.join(analysis.get('possible_entities', [])[:4]) or 'itens não classificados'}**.\n\n"
            "### Evidências encontradas\n"
            + "\n".join(ev_lines)
            + "\n\n### Próximo passo recomendado\n- Consolidar regras, exceções e oportunidades de automação a partir dessas evidências."
        )

    def _readiness_score(self, total_tables: int, total_chars: int, rules: int, exceptions: int) -> int:
        score = 40
        score += min(20, total_tables * 4)
        score += 10 if total_chars > 5000 else 0
        score += 10 if rules > 0 else 0
        score += 10 if exceptions > 0 else 0
        return min(95, score)

    def _build_priority_recommendations(self, total_tables: int, total_chars: int, entities: list[str]) -> list[dict[str, str]]:
        recs = [
            {"name": "Base RAG com documentos e regras", "impact": "Alto", "effort": "Médio", "priority": "Alta"},
            {"name": "Classificação e entendimento de documentos", "impact": "Alto", "effort": "Baixo", "priority": "Alta"},
        ]
        if total_tables > 0:
            recs.append({"name": "Consulta tabular dinâmica e filtros analíticos", "impact": "Alto", "effort": "Médio", "priority": "Alta"})
        if any(e in entities for e in ["incidente", "cliente", "contrato"]):
            recs.append({"name": "Extração estruturada de campos-chave", "impact": "Médio", "effort": "Médio", "priority": "Média"})
        return recs[:4]

    def _recommended_capabilities(self, analysis: dict[str, Any]) -> list[str]:
        capabilities = ["RAG", "resumo", "workflow"]
        if analysis.get("total_tables", 0) > 0:
            capabilities.extend(["analytics tabular", "filtros dinâmicos"])
        if analysis.get("rules_detected"):
            capabilities.append("classificação")
        return list(dict.fromkeys(capabilities))

    def _infer_objective(self, documents: list[dict[str, Any]], entities: list[str]) -> str:
        names = ", ".join([d.get("filename", "arquivo") for d in documents[:3]])
        ent = ", ".join(entities[:4]) if entities else "dados operacionais"
        return f"Entender o processo e o teor dos materiais ({names}) para estruturar conhecimento sobre {ent} e orientar automação." 

    def _infer_bottlenecks(self, analysis: dict[str, Any]) -> list[str]:
        items = ["Dependência de leitura manual e interpretação de documentos"]
        if analysis.get("total_tables", 0) > 0:
            items.append("Necessidade de filtrar e consolidar dados tabulares antes da execução")
        if analysis.get("warnings"):
            items.append("Qualidade de extração desigual em parte dos arquivos")
        return items

    def _infer_risks(self, analysis: dict[str, Any]) -> list[str]:
        items = ["Regras implícitas podem não estar totalmente padronizadas"]
        if analysis.get("exceptions_detected"):
            items.append("Existem indícios de exceções operacionais e inconsistências")
        if analysis.get("total_tables", 0) > 0:
            items.append("Perguntas quantitativas exigem consulta estruturada para evitar contagens imprecisas")
        return items

    def _priority_label(self, score: int) -> str:
        if score >= 75:
            return "alta"
        if score >= 55:
            return "média"
        return "baixa"

    def _diagnostic_summary(self, analysis: dict[str, Any], suggestions: list[str], capabilities: list[str]) -> str:
        return (
            f"O caso apresenta prontidão estimada de {analysis.get('automation_readiness_score', 0)}% para automação inicial. "
            f"As recomendações mais relevantes são: {', '.join(suggestions[:3]) or 'sem recomendações suficientes'}. "
            f"As capacidades recomendadas incluem: {', '.join(capabilities[:4])}."
        )

    def _interesting_lines(self, text: str) -> list[str]:
        lines = re.split(r"[\r\n]+", text or "")
        cleaned = [line.strip() for line in lines if line.strip()]
        return cleaned[:300]

    def _clip(self, text: str, limit: int = 180) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
