from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.repositories.json_case_repository import JsonCaseRepository
from app.services.analysis import AnalysisService
from app.services.automation import AutomationService
from app.services.graph import AnalysisGraphService
from app.services.parsers import ParserService
from app.services.retrieval import RetrievalService

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="GABBI Enterprise V7 Wizard", version="7.0.0", description="Discovery + entendimento + automação assistida com analytics tabular")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

repo = JsonCaseRepository(base_path=DATA_DIR)
parser_service = ParserService()
retrieval_service = RetrievalService()
analysis_service = AnalysisService()
graph_service = AnalysisGraphService(retrieval_service=retrieval_service, analysis_service=analysis_service)
automation_service = AutomationService()


class CreateCaseRequest(BaseModel):
    name: str = Field(...)
    description: str | None = None


class AskRequest(BaseModel):
    question: str
    mode: str = "executive"


@app.get("/", response_model=None)
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"name": "GABBI Enterprise V7 Wizard", "status": "ok", "message": "Frontend não encontrado."}


@app.get("/health")
async def health():
    return {"status": "ok", "llm": graph_service.llm_status(), "vector": retrieval_service.status()}


@app.get("/llm/status")
async def llm_status():
    return graph_service.llm_status()


@app.get("/vector/status")
async def vector_status():
    return retrieval_service.status()


@app.post("/cases")
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
        "specialist_state": None,
    }
    repo.create_case(case_id, case_data)
    return {"case_id": case_id, "message": "Caso criado com sucesso"}


@app.get("/cases")
async def list_cases():
    return {"items": repo.list_cases()}


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    payload = repo.get_case(case_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Case not found")
    return payload


@app.post("/cases/{case_id}/upload")
async def upload_files(case_id: str, files: list[UploadFile] = File(...)):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    uploaded = []
    for upload in files:
        filename = upload.filename or f"arquivo_{uuid4().hex[:6]}"
        content = await upload.read()
        saved_path = repo.save_uploaded_file(case_id, filename, content)
        parsed = parser_service.parse_file(saved_path)
        document = {"id": uuid4().hex[:12], "filename": filename, "path": saved_path, "content_type": upload.content_type, "parsed": parsed}
        repo.add_document(case_id, document)
        uploaded.append({"filename": filename, "path": saved_path, "content_type": upload.content_type, "text_length": len(parsed.get("text", "") or ""), "tables_found": len(parsed.get("tables", []) or [])})
    case = repo.get_case(case_id)
    documents = case.get("documents", [])
    publication = retrieval_service.build_case_index(case_id, documents)
    analysis = analysis_service.generate_initial_analysis(documents)
    tabular_catalog = graph_service.build_tabular_catalog(case_id, documents)
    repo.update_case(case_id, {"analysis": analysis, "vector_publication": publication, "tabular_catalog": tabular_catalog})
    return {"case_id": case_id, "uploaded": uploaded, "analysis": analysis, "vector_publication": publication, "tabular_catalog": tabular_catalog}


@app.get("/cases/{case_id}/analysis")
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


@app.get("/cases/{case_id}/tables/catalog")
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


@app.post("/cases/{case_id}/ask")
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
        specialist_state=case.get("specialist_state"),
    )
    if result.get("specialist_state") is not None:
        repo.update_case(case_id, {"specialist_state": result.get("specialist_state")})
    chat_item = {
        "id": uuid4().hex[:12],
        "question": payload.question,
        "mode": payload.mode,
        "route": result.get("route"),
        "query_type": result.get("query_type"),
        "answer_text": result.get("answer_text") or result.get("summary", ""),
        "evidence_files": result.get("evidence_files", []),
        "specialist_state": result.get("specialist_state"),
    }
    repo.append_chat_history(case_id, chat_item)
    return {"case_id": case_id, **result}






@app.post("/cases/{case_id}/specialist-state/reset")
async def reset_specialist_state(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    repo.update_case(case_id, {"specialist_state": None})
    return {"case_id": case_id, "message": "Estado do especialista resetado."}
@app.get("/cases/{case_id}/specialist-state")
async def get_specialist_state(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case_id": case_id, "specialist_state": case.get("specialist_state")}


@app.post("/cases/{case_id}/diagnostic")
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
    repo.update_case(case_id, {"analysis": analysis, "diagnostic": diagnostic, "blueprint": blueprint, "agent_config": agent_config})
    return {"case_id": case_id, "diagnostic": diagnostic, "blueprint": blueprint, "agent_config": agent_config}


@app.get("/cases/{case_id}/diagnostic")
async def get_diagnostic(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.get("diagnostic"):
        raise HTTPException(status_code=404, detail="Diagnostic not generated yet")
    return {"case_id": case_id, "diagnostic": case["diagnostic"], "blueprint": case.get("blueprint"), "agent_config": case.get("agent_config")}


@app.post("/cases/{case_id}/publish-vector")
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


@app.post("/cases/{case_id}/export/n8n")
async def export_n8n(case_id: str):
    case = repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.get("diagnostic"):
        raise HTTPException(status_code=400, detail="Generate diagnostic first")
    workflow = automation_service.build_n8n_workflow(case_id, case.get("analysis", {}), case.get("diagnostic", {}))
    content = json.dumps(workflow, ensure_ascii=False, indent=2)
    path = repo.save_export(case_id, "workflow_n8n", content, "json")
    repo.update_case(case_id, {"workflow_export": path})
    return {"case_id": case_id, "path": path, "workflow": workflow}


@app.get("/cases/{case_id}/export/n8n")
async def download_n8n(case_id: str):
    case = repo.get_case(case_id)
    if not case or not case.get("workflow_export"):
        raise HTTPException(status_code=404, detail="Workflow export not found")
    return FileResponse(case["workflow_export"], filename=f"gabbi_{case_id}_workflow.json", media_type="application/json")


@app.post("/cases/{case_id}/export/blueprint")
async def export_blueprint(case_id: str):
    case = repo.get_case(case_id)
    if not case or not case.get("blueprint"):
        raise HTTPException(status_code=404, detail="Blueprint not found")
    content = json.dumps(case["blueprint"], ensure_ascii=False, indent=2)
    path = repo.save_export(case_id, "blueprint", content, "json")
    return {"case_id": case_id, "path": path, "blueprint": case["blueprint"]}


@app.get("/cases/{case_id}/export/blueprint")
async def download_blueprint(case_id: str):
    target = Path(DATA_DIR / case_id / "exports" / "blueprint.json")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Blueprint export not found")
    return FileResponse(str(target), filename=f"gabbi_{case_id}_blueprint.json", media_type="application/json")
