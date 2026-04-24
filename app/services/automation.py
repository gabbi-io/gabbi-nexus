from __future__ import annotations

class AutomationService:
    def build_blueprint(self, case_id: str, analysis: dict, diagnostic: dict) -> dict:
        return {
            "case_id": case_id,
            "title": "Blueprint Inicial de Automação",
            "objective": diagnostic.get("objective_of_process"),
            "stages": [
                {"order": 1, "name": "Receber documentos e bases", "description": "Entrada de arquivos do processo para análise."},
                {"order": 2, "name": "Extrair e classificar conteúdo", "description": "Separar conteúdo útil, metadados, tabelas e tipos de documento."},
                {"order": 3, "name": "Consolidar regras e exceções", "description": "Mapear regras de decisão e casos ambíguos."},
                {"order": 4, "name": "Responder perguntas com evidência", "description": "Permitir análise assistida com RAG e conversa contextual."},
                {"order": 5, "name": "Executar automação assistida", "description": "Acionar workflow n8n, extrações e validação humana."},
            ],
            "entities": diagnostic.get("main_entities", []),
            "rules": diagnostic.get("decision_rules", []),
            "exceptions": diagnostic.get("exceptions", []),
            "recommendations": diagnostic.get("automation_suggestions", []),
            "recommended_capabilities": diagnostic.get("recommended_capabilities", []),
            "readiness_score": diagnostic.get("automation_readiness_score"),
        }

    def build_n8n_workflow(self, case_id: str, analysis: dict, diagnostic: dict) -> dict:
        return {
            "name": f"GABBI - Caso {case_id}",
            "nodes": [
                {"parameters": {}, "id": "ManualTrigger_1", "name": "Manual Trigger", "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1, "position": [200, 300]},
                {"parameters": {"mode": "runOnceForAllItems", "jsCode": "return [{json:{case_id:'%s', objective:'%s', priority:'%s'}}];" % (case_id, (diagnostic.get('objective_of_process') or '').replace("'", "\'"), str(diagnostic.get('priority_level') or ''))}, "id": "Code_1", "name": "Preparar Contexto", "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [460, 300]},
                {"parameters": {"conditions": {"string": [{"value1": "={{$json.priority}}", "operation": "contains", "value2": "alta"}]}}, "id": "If_1", "name": "Prioridade Alta?", "type": "n8n-nodes-base.if", "typeVersion": 2, "position": [720, 300]},
                {"parameters": {"assignments": {"assignments": [{"name": "action", "value": "Gerar base RAG + extração de campos + validação humana", "type": "string"}]}}, "id": "Set_1", "name": "Plano Alta Prioridade", "type": "n8n-nodes-base.set", "typeVersion": 3.4, "position": [980, 220]},
                {"parameters": {"assignments": {"assignments": [{"name": "action", "value": "Gerar classificação documental + checklist de validação", "type": "string"}]}}, "id": "Set_2", "name": "Plano Padrão", "type": "n8n-nodes-base.set", "typeVersion": 3.4, "position": [980, 380]},
                {"parameters": {"assignments": {"assignments": [{"name": "summary", "value": diagnostic.get("executive_summary", ""), "type": "string"}, {"name": "capabilities", "value": ", ".join(diagnostic.get("recommended_capabilities", [])), "type": "string"}]}}, "id": "Set_3", "name": "Consolidar Saída", "type": "n8n-nodes-base.set", "typeVersion": 3.4, "position": [1240, 300]}
            ],
            "connections": {
                "Manual Trigger": {"main": [[{"node": "Preparar Contexto", "type": "main", "index": 0}]]},
                "Preparar Contexto": {"main": [[{"node": "Prioridade Alta?", "type": "main", "index": 0}]]},
                "Prioridade Alta?": {"main": [[{"node": "Plano Alta Prioridade", "type": "main", "index": 0}], [{"node": "Plano Padrão", "type": "main", "index": 0}]]},
                "Plano Alta Prioridade": {"main": [[{"node": "Consolidar Saída", "type": "main", "index": 0}]]},
                "Plano Padrão": {"main": [[{"node": "Consolidar Saída", "type": "main", "index": 0}]]}
            },
            "pinData": {},
            "meta": {"templateCredsSetupCompleted": True, "generatedBy": "GABBI Enterprise V4", "caseId": case_id},
            "settings": {"executionOrder": "v1"},
            "tags": ["gabbi", "discovery", "automation"],
            "versionId": "1"
        }

    def build_agent_config(self, diagnostic: dict) -> dict:
        return {
            "name": "Agente de Entendimento de Processo",
            "persona": "Analista de automação corporativa orientado a evidências",
            "instructions": [
                "Sempre responder com base nas evidências recuperadas.",
                "Separar fatos observados, inferências e recomendações.",
                "Sinalizar ambiguidades e solicitar validação humana quando necessário.",
            ],
            "tools": ["busca vetorial", "consulta de documentos", "geração de resumo", "geração de blueprint"],
            "recommended_capabilities": diagnostic.get("recommended_capabilities", []),
        }
