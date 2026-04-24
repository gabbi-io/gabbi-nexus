from __future__ import annotations

import re
from collections import Counter
from typing import Any

class AnalysisService:
    COMMON_ENTITY_HINTS = {
        "cliente": ["cliente", "beneficiário", "usuario", "usuário", "consumidor"],
        "contrato": ["contrato", "apólice", "adesão", "plano"],
        "pedido": ["pedido", "solicitação", "requisição", "protocolo"],
        "produto": ["produto", "pacote", "serviço", "oferta"],
        "data": ["data", "vigência", "competência", "prazo"],
        "valor": ["valor", "preço", "tarifa", "custo"],
        "status": ["status", "situação", "estado"],
    }

    def generate_initial_analysis(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        filenames, warnings, findings, rules, exceptions = [], [], [], [], []
        total_text_chars = 0
        total_tables = 0
        entity_counter = Counter()
        for doc in documents:
            filenames.append(doc.get("filename"))
            parsed = doc.get("parsed", {})
            text = parsed.get("text", "") or ""
            total_text_chars += len(text)
            tables = parsed.get("tables", []) or []
            total_tables += len(tables)
            low = text.lower()
            for entity, hints in self.COMMON_ENTITY_HINTS.items():
                if any(h in low for h in hints):
                    entity_counter[entity] += 1
            rules.extend(self._extract_rules(text))
            exceptions.extend(self._extract_exceptions(text))
            if len(text.strip()) < 100:
                warnings.append(f"Baixo volume de texto em {doc.get('filename')}")
            if text.count("Página") > 10 and len(text.strip()) < 200:
                warnings.append(f"Possível PDF com baixa extração textual em {doc.get('filename')}")
        findings.append(f"Total de tabelas identificadas: {total_tables}")
        findings.append(f"Volume total de texto extraído: {total_text_chars} caracteres")
        if entity_counter:
            findings.append("Entidades mais prováveis: " + ", ".join([k for k, _ in entity_counter.most_common(5)]))
        readiness_score = self._readiness_score(total_text_chars, total_tables, len(rules), len(exceptions), warnings)
        return {
            "summary": f"Foram analisados {len(documents)} arquivo(s) com foco em entendimento pré-automação. O material sugere oportunidade de organizar conhecimento, classificar documentos e extrair campos antes da automação.",
            "documents_count": len(documents),
            "filenames": filenames,
            "total_text_chars": total_text_chars,
            "total_tables": total_tables,
            "possible_entities": [k for k, _ in entity_counter.most_common(8)],
            "findings": findings,
            "rules_detected": rules[:20],
            "exceptions_detected": exceptions[:20],
            "warnings": warnings,
            "automation_readiness_score": readiness_score,
            "priority_recommendations": self._priority_recommendations(total_tables, total_text_chars, rules, exceptions),
        }

    def generate_structured_diagnostic(self, documents: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
        rules = analysis.get("rules_detected", [])
        exceptions = analysis.get("exceptions_detected", [])
        entities = analysis.get("possible_entities", [])
        score = analysis.get("automation_readiness_score", 62)
        recommendations = []
        if analysis.get("total_text_chars", 0) > 5000:
            recommendations.append("Criar base de conhecimento RAG com chunks revisados e metadados por documento.")
        if analysis.get("total_tables", 0) > 0:
            recommendations.append("Implementar extração estruturada de campos e classificação de linhas/abas.")
        recommendations.append("Adicionar checklist de validação humana para regras ambíguas e exceções.")
        recommendations.append("Gerar workflow inicial no n8n com etapa de triagem, extração, decisão e validação.")
        return {
            "objective_of_process": "Entender os dados e documentos, consolidar regras e exceções e preparar automação assistida.",
            "inputs": [doc.get("filename") for doc in documents],
            "outputs": ["Resumo executivo", "Achados principais", "Regras identificadas", "Exceções e inconsistências", "Sugestões de automação", "Blueprint do processo", "Workflow n8n inicial"],
            "main_entities": entities,
            "decision_rules": rules[:15] or ["As regras ainda parecem majoritariamente implícitas e devem ser refinadas com validação humana."],
            "exceptions": exceptions[:15] or ["Nenhuma exceção evidente foi encontrada de forma determinística; requer interpretação assistida."],
            "bottlenecks": ["Leitura manual de documentos extensos", "Regras espalhadas em texto livre", "Dependência de validação humana em casos ambíguos"],
            "risks": ["Campos críticos ausentes ou inconsistentes", "Regras de negócio implícitas e não documentadas", "Baixa padronização entre documentos"],
            "automation_suggestions": recommendations,
            "recommended_capabilities": ["RAG para perguntas sobre documentos e bases", "Classificação de documentos", "Extração estruturada de campos", "Workflow n8n com validação humana", "Agente de apoio à análise"],
            "automation_readiness_score": score,
            "priority_level": self._priority_label(score),
            "executive_summary": self._executive_summary(analysis, entities),
        }

    def format_answer(self, question: str, evidences: list[dict[str, Any]], analysis: dict[str, Any], mode: str = "executive") -> dict[str, Any]:
        evidence_files = sorted({e.get("filename") for e in evidences if e.get("filename")})
        bullets = [
            "Volume elevado de texto livre, exigindo classificação e segmentação antes da execução.",
            "Boa oportunidade para organizar conhecimento e regras em base consultável.",
            "Casos ambíguos devem seguir com validação humana.",
        ]
        if analysis.get("total_tables", 0) > 0:
            bullets.append("Existem tabelas/planilhas que favorecem extração estruturada de campos.")
        text = "Os dados indicam uma boa oportunidade para começar por organização do conhecimento, classificação de documentos e extração de campos, deixando decisões ambíguas com validação humana. A prioridade inicial recomendada é: criar base de conhecimento RAG com documentos e regras identificadas."
        payload = {
            "title": f"Resposta para: {question[:90]}",
            "summary": text,
            "bullets": bullets,
            "what_data_shows": [
                f"Total de arquivos analisados: {analysis.get('documents_count', 0)}.",
                f"Volume total de texto extraído: {analysis.get('total_text_chars', 0)} caracteres.",
                f"Entidades prováveis: {', '.join(analysis.get('possible_entities', [])[:6]) or 'não identificadas claramente' }.",
            ],
            "recommended_first_automation": ["Criar base de conhecimento RAG", "Classificar documentos por tipo/tema", "Extrair campos-chave para apoio à decisão"],
            "evidence_files": evidence_files,
            "evidences": evidences,
            "mode": mode,
        }
        if mode == "technical":
            payload["technical"] = {"analysis": analysis, "evidences": evidences}
        return payload

    def _extract_rules(self, text: str) -> list[str]:
        patterns = [r"se\s+.+?então.+?(?:\.|;)", r"dever[aá]\s+.+?(?:\.|;)", r"premissa[:\s].+?(?:\.|;)", r"resultado esperado[:\s].+?(?:\.|;)"]
        found = []
        norm = " ".join(text.split())
        for p in patterns:
            found.extend(re.findall(p, norm, flags=re.IGNORECASE))
        cleaned = []
        seen = set()
        for item in found:
            item = item.strip()
            if len(item) > 25 and item.lower() not in seen:
                cleaned.append(item[:280])
                seen.add(item.lower())
        return cleaned

    def _extract_exceptions(self, text: str) -> list[str]:
        markers = ["exceção", "erro", "caso ambíguo", "não encontrado", "inválido", "falha", "inconsist"]
        lines = re.split(r'[\.]+', text)
        out = []
        seen = set()
        for line in lines:
            low = line.lower().strip()
            if any(m in low for m in markers) and len(low) > 12 and low not in seen:
                out.append(line.strip()[:280])
                seen.add(low)
        return out

    def _readiness_score(self, total_text: int, total_tables: int, rules: int, exceptions: int, warnings: list[str]) -> int:
        score = 55
        if total_text > 5000:
            score += 10
        if total_tables > 0:
            score += 10
        if rules > 3:
            score += 10
        if exceptions > 0:
            score += 5
        score -= min(15, len(warnings) * 5)
        return max(35, min(95, score))

    def _priority_recommendations(self, total_tables: int, total_text: int, rules: list[str], exceptions: list[str]) -> list[dict[str, Any]]:
        recs = [
            {"name": "Base RAG", "impact": "alto", "effort": "baixo", "priority": "alta"},
            {"name": "Classificação de documentos", "impact": "alto", "effort": "médio", "priority": "alta"},
            {"name": "Checklist de validação humana", "impact": "médio", "effort": "baixo", "priority": "alta"},
        ]
        if total_tables > 0:
            recs.append({"name": "Extração estruturada de campos", "impact": "alto", "effort": "médio", "priority": "alta"})
        if len(rules) > 5:
            recs.append({"name": "Motor de decisão / regras", "impact": "alto", "effort": "alto", "priority": "média"})
        if exceptions:
            recs.append({"name": "Tratamento de exceções", "impact": "médio", "effort": "médio", "priority": "média"})
        return recs

    def _executive_summary(self, analysis: dict[str, Any], entities: list[str]) -> str:
        return f"O material analisado sugere um processo com forte dependência de leitura manual, regras dispersas em texto livre e oportunidade clara de estruturar conhecimento antes da execução. As entidades mais prováveis são: {', '.join(entities[:5]) or 'a confirmar'}. A recomendação é iniciar por RAG, classificação documental e extração estruturada com validação humana."

    def _priority_label(self, score: int) -> str:
        if score >= 80:
            return "alta"
        if score >= 65:
            return "média-alta"
        if score >= 50:
            return "média"
        return "baixa"
