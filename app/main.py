from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.repositories.json_case_repository import JsonCaseRepository
from app.repositories.gabbi_postgres_repository import GabbiPostgresRepository
from app.services.analysis import AnalysisService
from app.services.automation import AutomationService
from app.services.graph import AnalysisGraphService
from app.services.gabbi_chat_ingestion import GabbiChatIngestionService
from app.services.parsers import ParserService
from app.services.retrieval import RetrievalService

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"


OPENAPI_DESCRIPTION = """
# Gabbi Nexus — BFF Inteligente para IA Corporativa, RAG, ML e Automação

O **Gabbi Nexus** é uma plataforma de inteligência artificial corporativa de nova geração e atua como um **BFF Inteligente**
(**Backend for Frontend / Backend for AI**) plugável em qualquer sistema consumidor, incluindo o **Gabbi atual**, portais internos,
frontends corporativos, canais digitais, automações n8n, CRMs, ERPs, ServiceNow, intranets ou aplicações móveis.

Ele foi criado para complementar o Gabbi atual, reduzindo alucinações, respostas inconsistentes e perda de contexto por meio de
uma arquitetura com **LangChain**, **LangGraph**, **Machine Learning**, **banco vetorial**, **consulta tabular dinâmica** e
camadas internas de treinamento/fine-tuning.

## O problema no modelo tradicional

No modelo tradicional, a interação costuma seguir um fluxo simples:

```text
Usuário → Prompt → LLM → Resposta
```

Esse modelo é rápido para protótipos, mas pode gerar:

- respostas inconsistentes;
- alucinações;
- baixa rastreabilidade;
- perda de contexto;
- dificuldade de responder perguntas quantitativas;
- dependência excessiva de prompt;
- dificuldade de especializar a IA por domínio;
- pouca governança sobre aprendizado, evidências e automação.

## O que muda com o Gabbi Nexus

O **Gabbi Nexus** introduz uma camada intermediária inteligente entre os sistemas consumidores e os modelos de IA:

```text
Sistema Consumidor / Gabbi atual / Portal / App
        ↓
Gabbi Nexus como BFF Inteligente
        ↓
LangGraph Orchestrator
        ↓
LangChain Pipelines
        ↓
Banco Vetorial + Dados Corporativos + Consulta Tabular
        ↓
Machine Learning Layer
        ↓
LLM com contexto validado
        ↓
Resposta confiável, rastreável e pronta para automação
```

## Papel como BFF

Como BFF, o Gabbi Nexus não substitui necessariamente o Gabbi atual. Ele pode ser acoplado como uma camada especializada para:

- enriquecer respostas;
- recuperar evidências;
- validar contexto;
- classificar documentos;
- calcular scores de risco/inconsistência;
- orquestrar agentes;
- transformar conversas em diagnósticos;
- gerar blueprints e workflows;
- fornecer APIs reutilizáveis para qualquer frontend ou sistema.

## Camada de Machine Learning

A solução possui uma camada dedicada de **Machine Learning**. Hoje, o modelo principal previsto é o **Random Forest (RF)**,
especialmente útil para classificação, scoring e priorização. Porém, a arquitetura já prevê expansão para outros modelos:

- Random Forest;
- Gradient Boosting;
- XGBoost / LightGBM;
- modelos supervisionados por domínio;
- modelos de clusterização;
- modelos de detecção de anomalia;
- modelos híbridos combinando ML clássico + LLM;
- modelos especialistas treinados por área.

## H1 e H2 — Hidden Areas

A arquitetura prevê duas áreas internas ocultas, não necessariamente expostas ao usuário final:

### H1 — Hidden Training Area

Área reservada para preparação e treinamento controlado:

- curadoria de documentos;
- validação de datasets;
- rotulagem de exemplos;
- treinamento de classificadores;
- avaliação de precisão;
- criação de features;
- histórico de aprendizado;
- versionamento de modelos.

### H2 — Hidden Fine-Tuning Area

Área reservada para especialização e refinamento:

- fine-tuning de modelos;
- ajuste de prompts e policies;
- avaliação de respostas;
- feedback humano;
- calibração por domínio;
- criação de agentes especialistas;
- melhoria contínua com base nas interações.

## Jornada recomendada de uso

1. `POST /cases` — cria o caso de análise.
2. `POST /cases/{case_id}/upload` — envia documentos.
3. `GET /cases/{case_id}/analysis` — consulta a análise inicial.
4. `GET /cases/{case_id}/tables/catalog` — verifica tabelas detectadas.
5. `POST /cases/{case_id}/ask` — conversa com os dados.
6. `POST /cases/{case_id}/diagnostic` — gera diagnóstico e blueprint.
7. `POST /cases/{case_id}/export/n8n` — gera workflow n8n.
8. `GET /cases/{case_id}/export/n8n` — baixa o JSON do workflow.

## Resultado esperado

O Gabbi Nexus entrega respostas mais confiáveis porque não depende apenas do LLM. Ele combina:

- evidências documentais;
- busca semântica;
- consulta tabular;
- roteamento inteligente;
- classificação por ML;
- análise estruturada;
- automação assistida;
- governança de treinamento e fine-tuning.
"""


TAGS_METADATA = [
    {
        "name": "00. Portal",
        "description": "Rotas de interface web e documentação customizada da solução.",
    },
    {
        "name": "01. Observabilidade",
        "description": "Rotas para verificar saúde da aplicação, status do LLM e status do índice vetorial.",
    },
    {
        "name": "02. Casos",
        "description": "Gestão dos casos de análise. Um caso representa uma investigação documental isolada.",
    },
    {
        "name": "03. Documentos",
        "description": "Upload, parsing, extração de texto/tabelas, análise inicial e indexação vetorial.",
    },
    {
        "name": "04. Análise e Catálogo",
        "description": "Consulta da análise inicial e do catálogo tabular detectado nos arquivos enviados.",
    },
    {
        "name": "05. Chat Inteligente",
        "description": "Perguntas em linguagem natural com roteamento entre RAG documental e consulta tabular dinâmica.",
    },
    {
        "name": "06. Diagnóstico",
        "description": "Geração e consulta de diagnóstico estruturado, blueprint e configuração inicial de agente.",
    },
    {
        "name": "07. Vetorial",
        "description": "Publicação/reprocessamento do índice vetorial do caso.",
    },
    {
        "name": "08. Exportações",
        "description": "Geração e download de artefatos como workflow n8n e blueprint JSON.",
    },
    {
        "name": "09. Integrações Gabbi",
        "description": "Endpoints para integração direta com o Gabbi, usando conversationId como chave externa e Article como base de conhecimento.",
    },
]


app = FastAPI(
    title="Gabbi Nexus",
    version="5.0.0",
    description=OPENAPI_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    docs_url=None,
    redoc_url=None,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "syntaxHighlight.theme": "obsidian",
    },
    contact={
        "name": "Gabbi Nexus / Spread",
        "url": "https://www.spread.com.br",
    },
    license_info={
        "name": "Proprietary / Internal Use",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

repo = JsonCaseRepository(base_path=DATA_DIR)
parser_service = ParserService()
retrieval_service = RetrievalService()
analysis_service = AnalysisService()
graph_service = AnalysisGraphService(retrieval_service=retrieval_service, analysis_service=analysis_service)
automation_service = AutomationService()


@lru_cache(maxsize=1)
def get_gabbi_chat_ingestion_service() -> GabbiChatIngestionService:
    """Cria a integração com o PostgreSQL do Gabbi sob demanda."""
    repository = GabbiPostgresRepository()
    return GabbiChatIngestionService(repository=repository)


class CreateCaseRequest(BaseModel):
    """Payload para criação de um novo caso de análise documental."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=120,
        description="Nome amigável do caso. Use um nome que identifique o objetivo da análise.",
        examples=["Análise de contratos de fornecedores - Maio/2026"],
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Descrição opcional do contexto, objetivo, área solicitante ou hipótese de automação.",
        examples=[
            "Caso criado para avaliar contratos, identificar riscos, extrair vencimentos e sugerir automações para acompanhamento."
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Análise de contratos de fornecedores - Maio/2026",
                "description": "Avaliar contratos enviados pela área jurídica e identificar riscos, vencimentos e oportunidades de automação.",
            }
        }
    }


class AskRequest(BaseModel):
    """Payload para conversa inteligente com os documentos e tabelas do caso."""

    question: str = Field(
        ...,
        min_length=3,
        description="Pergunta em linguagem natural. Pode envolver resumo, comparação, contagem, filtro, risco ou recomendação.",
        examples=[
            "Quais contratos vencem nos próximos 90 dias?",
            "Quantos incidentes P5 existem no canal AURA?",
            "O que vale automatizar primeiro com base nos documentos enviados?",
        ],
    )
    mode: str = Field(
        default="executive",
        description="Modo de resposta desejado. `executive` resume para negócio, `analytical` aprofunda análise e `technical` detalha aspectos técnicos.",
        examples=["executive"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Analise o material e me diga o que vale automatizar primeiro.",
                "mode": "executive",
            }
        }
    }


class GabbiChatAskRequest(BaseModel):
    """Payload para pergunta vinda do Gabbi usando a conversa atual como chave externa."""

    conversation_id: str = Field(
        ...,
        min_length=1,
        description="Valor da coluna Chat.conversationId no banco do Gabbi. É a chave externa da conversa em andamento.",
        examples=["conv_123456"],
    )
    question: str = Field(
        ...,
        min_length=3,
        description="Pergunta feita pelo usuário no Gabbi.",
        examples=["Como faço uma transação SAP personalizada?"],
    )
    session_id: str | None = Field(default=None, description="Opcional. Valor da coluna Chat.sessionId.")
    topic_id: int | None = Field(default=None, description="Opcional. Filtro usando Article.topicId. Não é a chave da conversa.")
    mode: str = Field(default="executive", description="Modo de resposta: executive, analytical ou technical.")
    article_limit: int = Field(default=100, ge=1, le=5000, description="Quantidade máxima de artigos a ingerir.")
    updated_after: datetime | None = Field(default=None, description="Opcional. Ingestão incremental por Article.updatedOn.")
    force_reindex: bool = Field(default=False, description="Força releitura dos artigos e reconstrução do índice vetorial.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation_id": "conv_123456",
                "question": "Como faço uma transação SAP personalizada?",
                "topic_id": 10,
                "mode": "executive",
                "article_limit": 100,
                "force_reindex": False,
            }
        }
    }


def find_case_by_external_conversation_id(conversation_id: str) -> dict | None:
    """Localiza um caso Nexus previamente vinculado ao Chat.conversationId do Gabbi."""
    for item in repo.list_cases():
        case_id = item.get("id") or item.get("case_id")
        if not case_id:
            continue
        case = repo.get_case(case_id)
        if case and case.get("external_conversation_id") == conversation_id:
            return case
    return None


def create_gabbi_conversation_case(*, conversation_id: str, session_id: str | None, chat_metadata: dict, topic_id: int | None) -> dict:
    """Cria um case Nexus amarrado à conversa atual do Gabbi."""
    case_id = uuid4().hex[:10]
    case_data = {
        "id": case_id,
        "name": f"Gabbi Chat - {conversation_id}",
        "description": "Caso criado automaticamente a partir de uma conversa do Gabbi.",
        "documents": [],
        "analysis": None,
        "diagnostic": None,
        "chat_history": [],
        "vector_publication": None,
        "blueprint": None,
        "workflow_export": None,
        "agent_config": None,
        "tabular_catalog": None,
        "source": "gabbi_chat",
        "external_conversation_id": conversation_id,
        "external_session_id": session_id or chat_metadata.get("session_id"),
        "gabbi_chat_metadata": chat_metadata,
        "topic_id": topic_id,
    }
    repo.create_case(case_id, case_data)
    return case_data


def ensure_gabbi_articles_indexed(*, case: dict, conversation_id: str, topic_id: int | None, article_limit: int, updated_after: datetime | None, force_reindex: bool) -> tuple[dict, list[dict], dict, dict, dict]:
    """Garante que os artigos do Gabbi estejam no caso e no índice vetorial."""
    case_id = case["id"]
    existing_documents = case.get("documents", []) or []
    should_ingest = force_reindex or not existing_documents

    if should_ingest:
        gabbi_service = get_gabbi_chat_ingestion_service()
        documents_from_db = gabbi_service.build_documents_from_articles(
            conversation_id=conversation_id,
            topic_id=topic_id,
            limit=article_limit,
            updated_after=updated_after,
        )
        if not documents_from_db:
            raise HTTPException(status_code=404, detail="Nenhum artigo publicado e válido foi encontrado para a conversa/tópico informado.")
        if force_reindex:
            repo.update_case(case_id, {"documents": documents_from_db})
        else:
            for document in documents_from_db:
                repo.add_document(case_id, document)

    case = repo.get_case(case_id)
    documents = case.get("documents", []) or []
    if not documents:
        raise HTTPException(status_code=400, detail="No documents available after Gabbi ingestion")

    must_rebuild = force_reindex or not case.get("vector_publication") or not case.get("analysis")
    if must_rebuild:
        publication = retrieval_service.build_case_index(case_id, documents)
        analysis = analysis_service.generate_initial_analysis(documents)
        tabular_catalog = graph_service.build_tabular_catalog(case_id, documents)
        repo.update_case(case_id, {"analysis": analysis, "vector_publication": publication, "tabular_catalog": tabular_catalog, "topic_id": topic_id})
        case = repo.get_case(case_id)
    else:
        publication = case.get("vector_publication") or {}
        analysis = case.get("analysis") or {}
        tabular_catalog = case.get("tabular_catalog") or {}

    return case, documents, publication, analysis, tabular_catalog


def custom_swagger_html() -> str:
    """Página profissional de documentação com Swagger UI + Mermaid."""

    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gabbi Nexus | API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {
      --spread-purple: #5F259F;
      --spread-purple-2: #7A3CC2;
      --spread-orange: #FF7A00;
      --bg: #F6F4FB;
      --panel: #FFFFFF;
      --text: #2E2E38;
      --muted: #6E6390;
      --border: #E6E0F2;
      --soft: #FAF8FE;
      --dark: #1F1630;
      --success: #1B5E20;
      --warn: #9A4D00;
      --error: #B42318;
      --shadow: 0 18px 42px rgba(95, 37, 159, .14);
      --radius: 22px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(95, 37, 159, .14), transparent 28%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 42%, #F1ECFA 100%);
      color: var(--text);
    }

    .hero {
      background:
        linear-gradient(135deg, #241338 0%, #5F259F 58%, #FF7A00 140%);
      color: white;
      padding: 34px 42px 40px;
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -120px;
      top: -100px;
      width: 420px;
      height: 420px;
      border-radius: 50%;
      background: rgba(255, 122, 0, .18);
      filter: blur(2px);
    }

    .hero-inner {
      position: relative;
      z-index: 1;
      max-width: 1480px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }

    .brand-mark {
      width: 54px;
      height: 54px;
      border-radius: 18px;
      background: linear-gradient(135deg, #FFB547, #FF7A00);
      display: grid;
      place-items: center;
      font-weight: 900;
      color: #241338;
      box-shadow: 0 10px 30px rgba(255, 122, 0, .28);
    }

    .brand small {
      display: block;
      font-size: 12px;
      letter-spacing: .18em;
      text-transform: uppercase;
      opacity: .78;
      margin-top: 2px;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(30px, 4vw, 52px);
      letter-spacing: -0.04em;
      line-height: 1.02;
    }

    .hero p {
      max-width: 900px;
      font-size: 17px;
      line-height: 1.72;
      opacity: .92;
      margin: 18px 0 0;
    }

    .hero-actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 24px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 44px;
      padding: 12px 16px;
      border-radius: 14px;
      font-weight: 800;
      text-decoration: none;
      border: 1px solid rgba(255,255,255,.20);
      color: white;
      background: rgba(255,255,255,.12);
      backdrop-filter: blur(8px);
    }

    .btn.primary {
      background: linear-gradient(135deg, #FFB547, #FF7A00);
      color: #241338;
      border: none;
    }

    .status-card {
      width: 320px;
      background: rgba(255,255,255,.12);
      border: 1px solid rgba(255,255,255,.20);
      border-radius: 24px;
      padding: 18px;
      backdrop-filter: blur(10px);
    }

    .status-card h3 { margin: 0 0 14px; }
    .status-grid { display: grid; gap: 10px; }
    .status-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,.10);
    }

    .layout {
      max-width: 1480px;
      margin: -24px auto 0;
      padding: 0 24px 44px;
      position: relative;
      z-index: 2;
    }

    .section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 18px;
    }

    .section h2 {
      margin: 0 0 10px;
      color: var(--spread-purple);
      font-size: 22px;
      letter-spacing: -.02em;
    }

    .section p {
      margin: 0;
      color: var(--muted);
      line-height: 1.68;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }

    .card {
      border: 1px solid var(--border);
      background: linear-gradient(180deg, #FFFFFF, #FBF9FF);
      border-radius: 18px;
      padding: 18px;
      min-height: 138px;
    }

    .card .icon {
      width: 42px;
      height: 42px;
      border-radius: 14px;
      background: #F4F0FB;
      color: var(--spread-purple);
      display: grid;
      place-items: center;
      font-weight: 900;
      margin-bottom: 14px;
    }

    .card h3 { margin: 0 0 8px; font-size: 16px; }
    .card p { font-size: 13px; }

    .diagram-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }

    .diagram-box {
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--soft);
      padding: 18px;
      overflow: auto;
    }

    .diagram-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }

    .diagram-title h3 {
      margin: 0;
      color: var(--dark);
      font-size: 16px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      background: #FFF4E5;
      color: #8A4300;
    }

    .journey {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 8px;
      margin-top: 14px;
    }

    .journey-step {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      font-size: 12px;
      font-weight: 800;
      color: var(--spread-purple);
      min-height: 76px;
    }

    #swagger-ui {
      background: #FFFFFF;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 12px;
      overflow: hidden;
    }

    .swagger-ui .topbar { display: none; }
    .swagger-ui .info { margin: 24px 0; }
    .swagger-ui .info .title {
      color: var(--spread-purple);
      font-size: 30px;
    }

    .swagger-ui .scheme-container {
      background: #FAF8FE;
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: none;
    }

    .swagger-ui .opblock.opblock-get { border-color: #B9A5D8; background: rgba(95,37,159,.04); }
    .swagger-ui .opblock.opblock-post { border-color: #FFC078; background: rgba(255,122,0,.05); }
    .swagger-ui .opblock .opblock-summary-method {
      border-radius: 10px;
      min-width: 76px;
    }
    .swagger-ui .btn.execute {
      background: var(--spread-purple);
      border-color: var(--spread-purple);
    }

    .footer {
      text-align: center;
      padding: 30px 20px 42px;
      color: var(--muted);
      font-size: 13px;
    }

    @media(max-width: 1100px) {
      .hero-inner { grid-template-columns: 1fr; }
      .status-card { width: 100%; }
      .cards { grid-template-columns: repeat(2, 1fr); }
      .journey { grid-template-columns: repeat(2, 1fr); }
    }

    @media(max-width: 720px) {
      .hero { padding: 28px 20px; }
      .layout { padding: 0 14px 32px; }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>

<body>
  <header class="hero">
    <div class="hero-inner">
      <div>
        <div class="brand">
          <div class="brand-mark">G</div>
          <div>
            <strong>Gabbi Nexus</strong>
            <small>Spread • IA Corporativa • RAG • Automação</small>
          </div>
        </div>
        <h1>BFF Inteligente para IA Corporativa, RAG, ML e Automação Assistida</h1>
        <p>
          Documentação técnica e negocial do Gabbi Nexus. Esta página combina Swagger UI,
          arquitetura C4, comparação com o Gabbi atual e diagramas de sequência para explicar
          como o Nexus atua como BFF plugável, orquestrando LangChain, LangGraph, banco vetorial,
          Machine Learning, H1/H2, análise, diagnóstico e automação.
        </p>
        <div class="hero-actions">
          <a class="btn primary" href="#swagger-ui">Explorar endpoints</a>
          <a class="btn" href="/redoc">Abrir ReDoc</a>
          <a class="btn" href="/openapi.json">OpenAPI JSON</a>
          <a class="btn" href="/">Abrir aplicação</a>
        </div>
      </div>

      <aside class="status-card">
        <h3>Fluxo recomendado</h3>
        <div class="status-grid">
          <div class="status-item"><span>1. Criar caso</span><strong>POST /cases</strong></div>
          <div class="status-item"><span>2. Enviar arquivos</span><strong>POST /upload</strong></div>
          <div class="status-item"><span>3. Perguntar</span><strong>POST /ask</strong></div>
          <div class="status-item"><span>4. Diagnóstico</span><strong>POST /diagnostic</strong></div>
          <div class="status-item"><span>5. Exportar</span><strong>n8n / Blueprint</strong></div>
        </div>
      </aside>
    </div>
  </header>

  <main class="layout">
    <section class="section">
      <h2>Visão da solução</h2>
      <p>
        O Gabbi Nexus complementa o GABBI atual ao reduzir alucinações e inconsistências.
        Em vez de depender apenas de prompt direto para LLM, a solução cria um caso, processa documentos,
        extrai texto e tabelas, publica contexto no índice vetorial e usa um grafo de análise para decidir
        se a pergunta deve ser respondida por RAG documental, consulta tabular ou síntese analítica.
      </p>

      <div class="cards" style="margin-top:18px">
        <div class="card">
          <div class="icon">BFF</div>
          <h3>BFF Inteligente</h3>
          <p>Camada plugável entre Gabbi, portais, apps e sistemas corporativos consumidores.</p>
        </div>
        <div class="card">
          <div class="icon">RAG</div>
          <h3>LangChain + Vetorial</h3>
          <p>Recupera contexto real, evidências e trechos documentais antes da resposta do LLM.</p>
        </div>
        <div class="card">
          <div class="icon">LG</div>
          <h3>LangGraph</h3>
          <p>Orquestra agentes, mantém estado e decide a rota: RAG, tabular, ML ou automação.</p>
        </div>
        <div class="card">
          <div class="icon">ML</div>
          <h3>ML Layer</h3>
          <p>Hoje com Random Forest, preparada para XGBoost, LightGBM, anomalias e modelos especialistas.</p>
        </div>
        <div class="card">
          <div class="icon">H1</div>
          <h3>Hidden Training Area</h3>
          <p>Área oculta para curadoria, rotulagem, datasets, features, treino e versionamento.</p>
        </div>
        <div class="card">
          <div class="icon">H2</div>
          <h3>Hidden Fine-Tuning Area</h3>
          <p>Área oculta para feedback humano, ajuste fino, prompts, policies e agentes especialistas.</p>
        </div>
        <div class="card">
          <div class="icon">TAB</div>
          <h3>Consulta Tabular</h3>
          <p>Responde perguntas quantitativas sobre planilhas, filtros, agrupamentos e contagens.</p>
        </div>
        <div class="card">
          <div class="icon">n8n</div>
          <h3>Automação Assistida</h3>
          <p>Converte diagnóstico em blueprint, configuração de agente e workflow n8n.</p>
        </div>
      </div>

      <div class="journey">
        <div class="journey-step">Sistema consumidor</div>
        <div class="journey-step">BFF Nexus</div>
        <div class="journey-step">Upload / Parsing</div>
        <div class="journey-step">Vetorial / RAG</div>
        <div class="journey-step">ML Layer</div>
        <div class="journey-step">H1 / H2</div>
        <div class="journey-step">Chat com evidência</div>
        <div class="journey-step">Automação</div>
      </div>
    </section>

    <section class="section">
      <h2>Arquitetura C4 e sequência</h2>
      <p>
        Os diagramas abaixo são renderizados com Mermaid diretamente nesta documentação.
        Eles mostram a comparação entre o Gabbi atual e o Gabbi Nexus, além da arquitetura C4 e da sequência fim a fim.
        O objetivo é deixar claro que o Nexus funciona como BFF inteligente e pode ser plugado em qualquer sistema consumidor.
      </p>

      <div class="diagram-grid" style="margin-top:18px">
        <div class="diagram-box">
          <div class="diagram-title">
            <h3>Comparativo — Gabbi atual vs Gabbi Nexus</h3>
            <span class="pill">Evolução arquitetural</span>
          </div>
          <pre class="mermaid">
flowchart LR
  subgraph ATUAL["Gabbi atual"]
    AUSER["Usuario"]
    APROMPT["Prompt direto"]
    ALLM["LLM"]
    ARES["Resposta com maior risco de alucinacao"]
    AUSER --> APROMPT --> ALLM --> ARES
  end

  subgraph NEXUS["Gabbi Nexus"]
    NUSER["Usuario ou Sistema"]
    BFF["BFF Inteligente"]
    GRAPH["LangGraph Orchestrator"]
    CHAIN["LangChain Pipelines"]
    VEC["Banco Vetorial e Dados Corporativos"]
    ML["Machine Learning Layer"]
    H1["H1 Hidden Training Area"]
    H2["H2 Hidden Fine Tuning Area"]
    NLLM["LLM com contexto validado"]
    NRES["Resposta confiavel com evidencias"]
    NUSER --> BFF --> GRAPH --> CHAIN --> VEC --> ML --> NLLM --> NRES
    H1 --> ML
    H2 --> NLLM
    GRAPH --> H1
    GRAPH --> H2
  end
          </pre>
        </div>

        <div class="diagram-box">
          <div class="diagram-title">
            <h3>C4 — Container / Componentes principais</h3>
            <span class="pill">Mermaid renderizado</span>
          </div>
          <pre class="mermaid">
flowchart LR
  U["Usuario / Analista de Negocio"]
  FE["Frontend Web"]
  API["FastAPI - Gabbi Nexus"]
  REPO["JsonCaseRepository"]
  PARSER["ParserService"]
  VEC["RetrievalService - Banco Vetorial"]
  GRAPH["AnalysisGraphService - LangGraph"]
  ANALYSIS["AnalysisService"]
  AUTO["AutomationService"]
  LLM["LLM - OpenAI ou Azure OpenAI"]
  STORAGE["Data Directory - Uploads e Exports"]

  U --> FE
  FE --> API
  API --> REPO
  API --> PARSER
  API --> VEC
  API --> GRAPH
  API --> ANALYSIS
  API --> AUTO
  REPO --> STORAGE
  PARSER --> STORAGE
  VEC --> STORAGE
  GRAPH --> VEC
  GRAPH --> ANALYSIS
  GRAPH --> LLM
  ANALYSIS --> LLM
  AUTO --> STORAGE
          </pre>
        </div>

        <div class="diagram-box">
          <div class="diagram-title">
            <h3>Sequência — Upload, análise, chat e automação</h3>
            <span class="pill">Mermaid renderizado</span>
          </div>
          <pre class="mermaid">
sequenceDiagram
  autonumber
  actor U as Usuario
  participant SYS as Gabbi atual / Sistema consumidor
  participant BFF as Gabbi Nexus BFF
  participant API as FastAPI
  participant Repo as JsonCaseRepository
  participant Parser as ParserService
  participant Vec as Banco Vetorial
  participant Graph as LangGraph Orchestrator
  participant ML as ML Layer RF e modelos futuros
  participant H1 as H1 Training Area
  participant H2 as H2 Fine Tuning Area
  participant Analysis as AnalysisService
  participant Auto as AutomationService
  participant LLM as LLM

  U->>SYS: Inicia analise ou conversa
  SYS->>BFF: Chama API do Nexus
  BFF->>API: POST /cases
  API->>Repo: create_case()
  Repo-->>API: case_id
  API-->>BFF: case_id
  BFF-->>SYS: case_id

  U->>SYS: Envia documentos
  SYS->>BFF: Upload de arquivos
  BFF->>API: POST /cases/{case_id}/upload
  API->>Repo: save_uploaded_file()
  API->>Parser: parse_file()
  Parser-->>API: texto + tabelas
  API->>Vec: build_case_index()
  API->>Analysis: generate_initial_analysis()
  API->>Graph: build_tabular_catalog()
  API->>ML: classificar documentos e calcular scores
  ML->>H1: consultar datasets e features treinadas
  ML-->>API: classificacao + score
  API->>Repo: update_case()
  API-->>BFF: analise + vetorial + catalogo + ML

  U->>SYS: Pergunta em linguagem natural
  SYS->>BFF: POST /ask
  BFF->>API: POST /cases/{case_id}/ask
  API->>Graph: ask(question, mode, history)
  Graph->>Vec: recuperar evidencias
  Graph->>ML: avaliar risco / prioridade / inconsistencia
  Graph->>H2: aplicar ajustes finos e policies
  Graph->>LLM: gerar resposta com contexto validado
  LLM-->>Graph: resposta
  Graph-->>API: rota + resposta + evidencias + scores
  API->>Repo: append_chat_history()
  API-->>BFF: resposta rastreavel
  BFF-->>SYS: resposta para o canal consumidor

  U->>SYS: Solicita diagnostico e automacao
  SYS->>BFF: POST /diagnostic
  BFF->>API: POST /cases/{case_id}/diagnostic
  API->>Analysis: generate_structured_diagnostic()
  API->>Auto: build_blueprint() + build_agent_config()
  API->>Repo: update_case()
  API-->>BFF: diagnostico + blueprint + agente
  BFF-->>SYS: artefatos para uso no Gabbi ou outro sistema
          </pre>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Camada de Machine Learning, H1 e H2</h2>
      <p>
        O Gabbi Nexus possui uma camada explícita de Machine Learning. Hoje o modelo de referência é o Random Forest,
        utilizado para classificação documental, score de inconsistência, recomendação de prioridade e detecção de padrões.
        A arquitetura, porém, está preparada para outros modelos supervisionados, modelos de anomalia, clusterização,
        gradient boosting e abordagens híbridas combinando ML clássico com LLM.
      </p>

      <div class="cards" style="margin-top:18px">
        <div class="card">
          <div class="icon">RF</div>
          <h3>Random Forest atual</h3>
          <p>Classifica documentos, calcula scores e ajuda a priorizar riscos e oportunidades de automação.</p>
        </div>
        <div class="card">
          <div class="icon">+</div>
          <h3>Modelos futuros</h3>
          <p>Preparado para XGBoost, LightGBM, anomalias, clusterização e modelos especialistas por domínio.</p>
        </div>
        <div class="card">
          <div class="icon">H1</div>
          <h3>Treinamento oculto</h3>
          <p>Área H1 para datasets, curadoria, rotulagem, features, treino, validação e versionamento.</p>
        </div>
        <div class="card">
          <div class="icon">H2</div>
          <h3>Fine-tuning oculto</h3>
          <p>Área H2 para feedback humano, ajuste fino, policies, prompts e especialização de agentes.</p>
        </div>
      </div>
    </section>

    <section id="swagger-section" class="section">
      <h2>Catálogo de endpoints</h2>
      <p>
        Use os grupos abaixo para navegar pela API. Cada endpoint possui descrição de objetivo,
        quando usar, pré-condições, resposta esperada e exemplos de payload.
      </p>
    </section>

    <section id="swagger-ui"></section>
  </main>

  <footer class="footer">
    Gabbi Nexus • Spread • Documentação gerada com Swagger UI + Mermaid
  </footer>

  <script>
    mermaid.initialize({
      startOnLoad: true,
      securityLevel: "loose",
      htmlLabels: false,
      flowchart: { useMaxWidth: true, curve: "basis" },
      sequence: { useMaxWidth: true },
      theme: "base",
      themeVariables: {
        primaryColor: "#F4F0FB",
        primaryTextColor: "#2E2E38",
        primaryBorderColor: "#5F259F",
        lineColor: "#7A3CC2",
        secondaryColor: "#FFF4E5",
        tertiaryColor: "#FFFFFF",
        fontFamily: "Inter, Segoe UI, Arial"
      }
    });

    window.onload = function() {
      SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        docExpansion: "none",
        defaultModelsExpandDepth: -1,
        displayRequestDuration: true,
        filter: true,
        tryItOutEnabled: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>
"""


@app.get("/", response_model=None, tags=["00. Portal"], summary="Abrir frontend da aplicação")
async def root():
    """
    Abre o frontend web da solução, caso o arquivo `static/index.html` exista.

    Quando usar:
    - Para acessar a experiência visual do Gabbi Nexus.
    - Para operar a jornada de criação de caso, upload, chat, diagnóstico e exportação.

    Retorno:
    - `FileResponse` com o HTML do frontend.
    - JSON simples informando que o frontend não foi encontrado, caso a pasta `static` não exista.
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"name": "Gabbi Nexus", "status": "ok", "message": "Frontend não encontrado."}


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return HTMLResponse(custom_swagger_html())


@app.get("/redoc", include_in_schema=False)
async def redoc():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="Gabbi Nexus | ReDoc",
    )


@app.get(
    "/health",
    tags=["01. Observabilidade"],
    summary="Verificar saúde geral da API",
    description="""
Retorna o status geral da aplicação, incluindo status do LLM e do backend vetorial.

Quando usar:
- Health check de infraestrutura.
- Validação de deploy.
- Diagnóstico rápido antes de iniciar uma demonstração.
- Verificação para saber se o LLM está ativo ou em fallback.

Resposta esperada:
- `status`: estado geral da API.
- `llm`: informações do provedor/modelo de linguagem.
- `vector`: informações do mecanismo de recuperação vetorial.
""",
)
async def health():
    return {"status": "ok", "llm": graph_service.llm_status(), "vector": retrieval_service.status()}


@app.get(
    "/llm/status",
    tags=["01. Observabilidade"],
    summary="Consultar status do LLM",
    description="""
Retorna informações sobre a camada de LLM utilizada pelo grafo de análise.

Quando usar:
- Confirmar se o modelo está habilitado.
- Verificar qual modelo está configurado.
- Diagnosticar respostas em modo fallback.
""",
)
async def llm_status():
    return graph_service.llm_status()


@app.get(
    "/vector/status",
    tags=["01. Observabilidade"],
    summary="Consultar status do backend vetorial",
    description="""
Retorna o status do mecanismo de recuperação semântica.

Quando usar:
- Confirmar se o banco vetorial/índice local está disponível.
- Verificar se a recuperação vetorial está operante.
- Apoiar diagnóstico de respostas sem evidência.
""",
)
async def vector_status():
    return retrieval_service.status()


@app.post(
    "/cases",
    tags=["02. Casos"],
    summary="Criar um novo caso de análise",
    description="""
Cria um caso lógico para agrupar documentos, análise, histórico de chat, diagnóstico e exportações.

Quando usar:
- Sempre no início de uma nova análise.
- Antes de enviar documentos.
- Para separar contextos de clientes, demandas ou automações diferentes.

Pré-condições:
- Informar pelo menos o nome do caso.

Resultado:
- Retorna um `case_id`, que será usado nas próximas chamadas.
""",
    responses={
        200: {
            "description": "Caso criado com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "case_id": "a1b2c3d4e5",
                        "message": "Caso criado com sucesso",
                    }
                }
            },
        }
    },
)
async def create_case(payload: CreateCaseRequest):
    case_id = uuid4().hex[:10]
    case_data = {
        "id": case_id,
        "name": payload.name,
        "description": payload.description,
        "documents": [],
        "analysis": None,
        "diagnostic": None,
        "chat_history": [],
        "vector_publication": None,
        "blueprint": None,
        "workflow_export": None,
        "agent_config": None,
        "tabular_catalog": None,
    }
    repo.create_case(case_id, case_data)
    return {"case_id": case_id, "message": "Caso criado com sucesso"}


@app.get(
    "/cases",
    tags=["02. Casos"],
    summary="Listar casos existentes",
    description="""
Lista os casos já criados no repositório local.

Quando usar:
- Recuperar histórico de análises.
- Exibir lista de casos no frontend.
- Encontrar um `case_id` criado anteriormente.
""",
)
async def list_cases():
    return {"items": repo.list_cases()}


@app.get(
    "/cases/{case_id}",
    tags=["02. Casos"],
    summary="Consultar dados completos de um caso",
    description="""
Retorna o payload completo do caso, incluindo documentos, análise, histórico, diagnóstico,
blueprint, catálogo tabular e exportações já realizadas.

Quando usar:
- Debug técnico.
- Reidratar tela do frontend.
- Consultar o estado completo de uma análise.
""",
    responses={
        404: {"description": "Caso não encontrado."},
    },
)
async def get_case(case_id: str):
    payload = repo.get_case(case_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Case not found")
    return payload


@app.post(
    "/cases/{case_id}/upload",
    tags=["03. Documentos"],
    summary="Enviar documentos para análise",
    description="""
Recebe um ou mais arquivos, salva no repositório do caso, executa parsing, publica o índice vetorial,
gera análise inicial e monta o catálogo tabular.

Tipos esperados:
- PDF
- DOCX
- XLSX/XLSM/XLS
- CSV
- TXT
- Outros formatos suportados pelo `ParserService`

O que acontece internamente:
1. O arquivo é salvo no diretório do caso.
2. O `ParserService` extrai texto e tabelas.
3. O documento é registrado no `JsonCaseRepository`.
4. O `RetrievalService` cria/atualiza o índice vetorial do caso.
5. O `AnalysisService` gera uma análise inicial.
6. O `AnalysisGraphService` monta o catálogo tabular.

Quando usar:
- Depois de criar um caso com `POST /cases`.
- Sempre que novos documentos precisarem entrar no contexto.
""",
    responses={
        400: {"description": "Requisição inválida ou sem arquivo."},
        404: {"description": "Caso não encontrado."},
    },
)
async def upload_files(case_id: str, files: list[UploadFile] = File(..., description="Lista de arquivos a serem processados.")):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    uploaded = []
    for upload in files:
        filename = upload.filename or f"arquivo_{uuid4().hex[:6]}"
        content = await upload.read()
        saved_path = repo.save_uploaded_file(case_id, filename, content)
        parsed = parser_service.parse_file(saved_path)
        document = {
            "id": uuid4().hex[:12],
            "filename": filename,
            "path": saved_path,
            "content_type": upload.content_type,
            "parsed": parsed,
        }
        repo.add_document(case_id, document)
        uploaded.append(
            {
                "filename": filename,
                "path": saved_path,
                "content_type": upload.content_type,
                "text_length": len(parsed.get("text", "") or ""),
                "tables_found": len(parsed.get("tables", []) or []),
            }
        )
    case = repo.get_case(case_id)
    documents = case.get("documents", [])
    publication = retrieval_service.build_case_index(case_id, documents)
    analysis = analysis_service.generate_initial_analysis(documents)
    tabular_catalog = graph_service.build_tabular_catalog(case_id, documents)
    repo.update_case(
        case_id,
        {
            "analysis": analysis,
            "vector_publication": publication,
            "tabular_catalog": tabular_catalog,
        },
    )
    return {
        "case_id": case_id,
        "uploaded": uploaded,
        "analysis": analysis,
        "vector_publication": publication,
        "tabular_catalog": tabular_catalog,
    }


@app.get(
    "/cases/{case_id}/analysis",
    tags=["04. Análise e Catálogo"],
    summary="Consultar análise inicial do caso",
    description="""
Retorna a análise inicial gerada a partir dos documentos enviados.

A análise pode conter:
- resumo executivo;
- quantidade de documentos;
- quantidade de caracteres;
- tabelas detectadas;
- entidades;
- regras;
- exceções;
- riscos;
- recomendações iniciais;
- score de prontidão para automação.

Quando usar:
- Após o upload.
- Para alimentar a visão geral do frontend.
- Para entender rapidamente o teor dos documentos.
""",
    responses={
        400: {"description": "Nenhum documento enviado ainda."},
        404: {"description": "Caso não encontrado."},
    },
)
async def get_analysis(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    documents = case.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    analysis = case.get("analysis")
    if not analysis:
        analysis = analysis_service.generate_initial_analysis(documents)
        repo.update_case(case_id, {"analysis": analysis})
    return {"case_id": case_id, "analysis": analysis}


@app.get(
    "/cases/{case_id}/tables/catalog",
    tags=["04. Análise e Catálogo"],
    summary="Consultar catálogo tabular do caso",
    description="""
Monta e retorna o catálogo de tabelas identificadas nos documentos enviados.

Esse endpoint é essencial para reduzir respostas inconsistentes em perguntas quantitativas.

Exemplos de perguntas suportadas pelo catálogo:
- Quantos incidentes P5 existem?
- Quais registros pertencem ao canal AURA?
- Quantos itens por status?
- Quais linhas atendem a determinado filtro?

Quando usar:
- Após upload de planilhas ou documentos com tabelas.
- Antes de perguntas analíticas no chat.
- Para validar quais colunas e abas foram reconhecidas.
""",
    responses={
        400: {"description": "Nenhum documento enviado ainda."},
        404: {"description": "Caso não encontrado."},
    },
)
async def get_tabular_catalog(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    documents = case.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    catalog = graph_service.build_tabular_catalog(case_id, documents)
    repo.update_case(case_id, {"tabular_catalog": catalog})
    return {"case_id": case_id, "tabular_catalog": catalog}


@app.post(
    "/cases/{case_id}/ask",
    tags=["05. Chat Inteligente"],
    summary="Perguntar ao caso usando IA com evidências",
    description="""
Executa uma pergunta em linguagem natural sobre o material do caso.

O `AnalysisGraphService` decide a melhor rota:
- `documental/RAG`: quando a resposta depende de texto, cláusulas, regras e contexto documental.
- `tabular`: quando a pergunta envolve contagem, filtros, agrupamentos ou análise de planilhas.
- `analytical`: quando a pergunta exige síntese, diagnóstico ou recomendação.

O endpoint registra o histórico da conversa no caso.

Boas práticas:
- Prefira perguntas específicas.
- Para perguntas quantitativas, cite filtros esperados.
- Para respostas executivas, use `mode = executive`.
- Para respostas técnicas, use `mode = technical`.

Exemplos:
- "Quais contratos vencem nos próximos 90 dias?"
- "Quantos incidentes P5 existem no canal AURA?"
- "Explique os principais riscos encontrados nos documentos."
""",
    responses={
        400: {"description": "Nenhum documento enviado ainda."},
        404: {"description": "Caso não encontrado."},
    },
)
async def ask_case(case_id: str, payload: AskRequest):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    documents = case.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    if case.get("analysis") is None:
        analysis = analysis_service.generate_initial_analysis(documents)
        repo.update_case(case_id, {"analysis": analysis})
        case = repo.get_case(case_id)
    result = graph_service.ask(
        case_id=case_id,
        question=payload.question,
        analysis=case.get("analysis", {}),
        documents=documents,
        chat_history=case.get("chat_history", []),
        mode=payload.mode,
    )
    chat_item = {
        "id": uuid4().hex[:12],
        "question": payload.question,
        "mode": payload.mode,
        "route": result.get("route"),
        "query_type": result.get("query_type"),
        "answer_text": result.get("answer_text") or result.get("summary", ""),
        "evidence_files": result.get("evidence_files", []),
    }
    repo.append_chat_history(case_id, chat_item)
    return {"case_id": case_id, **result}


@app.post(
    "/integrations/gabbi/chat/ask",
    tags=["09. Integrações Gabbi"],
    summary="Perguntar ao Nexus usando a conversa atual do Gabbi",
    description="""
Endpoint para ser chamado pelo Gabbi durante uma conversa em andamento.

Chave externa adotada:
- `conversation_id` deve receber o valor de `Chat.conversationId`.

Papel dos demais campos:
- `topic_id` é filtro opcional sobre `Article.topicId`;
- `question` é a pergunta atual do usuário;
- `force_reindex` força nova leitura da tabela `Article` e reconstrução do índice vetorial.
""",
)
async def gabbi_chat_ask(payload: GabbiChatAskRequest):
    try:
        gabbi_service = get_gabbi_chat_ingestion_service()
        chat_metadata = gabbi_service.get_chat_metadata(payload.conversation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar o banco PostgreSQL do Gabbi: {exc}") from exc

    case = find_case_by_external_conversation_id(payload.conversation_id)
    if not case:
        case = create_gabbi_conversation_case(
            conversation_id=payload.conversation_id,
            session_id=payload.session_id,
            chat_metadata=chat_metadata,
            topic_id=payload.topic_id,
        )

    try:
        case, documents, publication, analysis, tabular_catalog = ensure_gabbi_articles_indexed(
            case=case,
            conversation_id=payload.conversation_id,
            topic_id=payload.topic_id,
            article_limit=payload.article_limit,
            updated_after=payload.updated_after,
            force_reindex=payload.force_reindex,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao ingerir artigos do Gabbi: {exc}") from exc

    result = graph_service.ask(
        case_id=case["id"],
        question=payload.question,
        analysis=case.get("analysis", analysis),
        documents=documents,
        chat_history=case.get("chat_history", []),
        mode=payload.mode,
    )

    chat_item = {
        "id": uuid4().hex[:12],
        "source": "gabbi_chat",
        "external_conversation_id": payload.conversation_id,
        "external_session_id": payload.session_id or chat_metadata.get("session_id"),
        "topic_id": payload.topic_id,
        "question": payload.question,
        "mode": payload.mode,
        "route": result.get("route"),
        "query_type": result.get("query_type"),
        "answer_text": result.get("answer_text") or result.get("summary", ""),
        "evidence_files": result.get("evidence_files", []),
    }
    repo.append_chat_history(case["id"], chat_item)

    return {
        "case_id": case["id"],
        "source": "gabbi_chat",
        "conversation_id": payload.conversation_id,
        "session_id": payload.session_id or chat_metadata.get("session_id"),
        "chat_found_in_gabbi": chat_metadata.get("found", False),
        "topic_id": payload.topic_id,
        "documents_available": len(documents),
        "vector_publication": publication,
        "tabular_catalog": tabular_catalog,
        **result,
    }


@app.post(
    "/cases/{case_id}/diagnostic",
    tags=["06. Diagnóstico"],
    summary="Gerar diagnóstico estruturado e blueprint",
    description="""
Gera diagnóstico estruturado do caso e cria artefatos de automação.

O que é produzido:
- Diagnóstico executivo.
- Riscos e gargalos.
- Sugestões de automação.
- Blueprint de automação.
- Configuração inicial de agente.

Quando usar:
- Após upload e análise inicial.
- Depois de fazer perguntas exploratórias.
- Quando o objetivo for transformar entendimento em automação.
""",
    responses={
        400: {"description": "Nenhum documento enviado ainda."},
        404: {"description": "Caso não encontrado."},
    },
)
async def generate_diagnostic(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    documents = case.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    analysis = case.get("analysis") or analysis_service.generate_initial_analysis(documents)
    diagnostic = analysis_service.generate_structured_diagnostic(documents, analysis)
    blueprint = automation_service.build_blueprint(case_id, analysis, diagnostic)
    agent_config = automation_service.build_agent_config(diagnostic)
    repo.update_case(
        case_id,
        {
            "analysis": analysis,
            "diagnostic": diagnostic,
            "blueprint": blueprint,
            "agent_config": agent_config,
        },
    )
    return {
        "case_id": case_id,
        "diagnostic": diagnostic,
        "blueprint": blueprint,
        "agent_config": agent_config,
    }


@app.get(
    "/cases/{case_id}/diagnostic",
    tags=["06. Diagnóstico"],
    summary="Consultar diagnóstico gerado",
    description="""
Retorna o diagnóstico estruturado previamente gerado.

Quando usar:
- Reabrir uma análise.
- Exibir diagnóstico no frontend.
- Consultar blueprint e configuração de agente sem regerar o diagnóstico.
""",
    responses={
        404: {"description": "Caso não encontrado ou diagnóstico ainda não gerado."},
    },
)
async def get_diagnostic(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.get("diagnostic"):
        raise HTTPException(status_code=404, detail="Diagnostic not generated yet")
    return {
        "case_id": case_id,
        "diagnostic": case["diagnostic"],
        "blueprint": case.get("blueprint"),
        "agent_config": case.get("agent_config"),
    }


@app.post(
    "/cases/{case_id}/publish-vector",
    tags=["07. Vetorial"],
    summary="Republicar índice vetorial do caso",
    description="""
Reprocessa os documentos do caso e atualiza o índice vetorial.

Quando usar:
- Após upload adicional.
- Quando houver necessidade de reconstruir o índice.
- Para validar se o RAG está usando o material mais recente.

Resultado:
- Metadados da publicação vetorial.
- Quantidade de chunks/documentos indexados, conforme implementação do `RetrievalService`.
""",
    responses={
        400: {"description": "Nenhum documento enviado ainda."},
        404: {"description": "Caso não encontrado."},
    },
)
async def publish_vector(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    documents = case.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    publication = retrieval_service.build_case_index(case_id, documents)
    repo.update_case(case_id, {"vector_publication": publication})
    return {"case_id": case_id, "vector_publication": publication}


@app.post(
    "/cases/{case_id}/export/n8n",
    tags=["08. Exportações"],
    summary="Gerar workflow n8n a partir do diagnóstico",
    description="""
Gera um workflow n8n importável com base na análise e no diagnóstico do caso.

Pré-condição:
- O diagnóstico precisa ter sido gerado via `POST /cases/{case_id}/diagnostic`.

Quando usar:
- Ao final da jornada de discovery.
- Quando a análise já identificou uma automação candidata.
- Para acelerar a criação de um fluxo operacional no n8n.
""",
    responses={
        400: {"description": "Diagnóstico ainda não gerado."},
        404: {"description": "Caso não encontrado."},
    },
)
async def export_n8n(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.get("diagnostic"):
        raise HTTPException(status_code=400, detail="Generate diagnostic first")
    workflow = automation_service.build_n8n_workflow(
        case_id,
        case.get("analysis", {}),
        case.get("diagnostic", {}),
    )
    content = json.dumps(workflow, ensure_ascii=False, indent=2)
    path = repo.save_export(case_id, "workflow_n8n", content, "json")
    repo.update_case(case_id, {"workflow_export": path})
    return {"case_id": case_id, "path": path, "workflow": workflow}


@app.get(
    "/cases/{case_id}/export/n8n",
    tags=["08. Exportações"],
    summary="Baixar workflow n8n gerado",
    description="""
Realiza o download do workflow n8n previamente exportado.

Pré-condição:
- Executar `POST /cases/{case_id}/export/n8n`.

Retorno:
- Arquivo `.json` importável no n8n.
""",
    responses={
        404: {"description": "Workflow exportado não encontrado."},
    },
)
async def download_n8n(case_id: str):
    case = repo.get_case(case_id)
    if not case or not case.get("workflow_export"):
        raise HTTPException(status_code=404, detail="Workflow export not found")
    return FileResponse(
        case["workflow_export"],
        filename=f"gabbi_{case_id}_workflow.json",
        media_type="application/json",
    )


@app.post(
    "/cases/{case_id}/export/blueprint",
    tags=["08. Exportações"],
    summary="Exportar blueprint de automação",
    description="""
Exporta o blueprint do caso em JSON.

Pré-condição:
- O blueprint deve existir, normalmente após `POST /cases/{case_id}/diagnostic`.

Quando usar:
- Para versionar a proposta de automação.
- Para integrar com outros sistemas.
- Para revisar tecnicamente a automação antes de implementar.
""",
    responses={
        404: {"description": "Caso ou blueprint não encontrado."},
    },
)
async def export_blueprint(case_id: str):
    case = repo.get_case(case_id)
    if not case or not case.get("blueprint"):
        raise HTTPException(status_code=404, detail="Blueprint not found")
    content = json.dumps(case["blueprint"], ensure_ascii=False, indent=2)
    path = repo.save_export(case_id, "blueprint", content, "json")
    return {"case_id": case_id, "path": path, "blueprint": case["blueprint"]}


@app.get(
    "/cases/{case_id}/export/blueprint",
    tags=["08. Exportações"],
    summary="Baixar blueprint exportado",
    description="""
Realiza o download do blueprint JSON previamente exportado.

Pré-condição:
- Executar `POST /cases/{case_id}/export/blueprint`.

Retorno:
- Arquivo `gabbi_{case_id}_blueprint.json`.
""",
    responses={
        404: {"description": "Blueprint exportado não encontrado."},
    },
)
async def download_blueprint(case_id: str):
    target = Path(DATA_DIR / case_id / "exports" / "blueprint.json")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Blueprint export not found")
    return FileResponse(
        str(target),
        filename=f"gabbi_{case_id}_blueprint.json",
        media_type="application/json",
    )
