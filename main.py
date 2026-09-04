import io
import os
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fallback placeholder to allow server startup if .env doesn't specify GROQ_API_KEY yet
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "gsk_placeholder_please_set_in_env"

# Import project backend modules
from agent.graph import build_agent
from rag.llm_rag import client

app = FastAPI(
    title="ClauseGuard / ContractRisk AI API",
    description="REST API wrapping RAG, LangGraph agent, and Groq LLM contract risk analysis",
    version="1.0.0",
)

# Enable CORS for frontend (development + production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for easy deployment on Vercel/Railway
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# HELPERS (Exact logic from app.py)
# ================================================================
def classify_risk(analysis_text: str) -> str:
    txt = analysis_text.lower()
    if "high" in txt:
        return "High"
    elif "medium" in txt:
        return "Medium"
    return "Low"


def generate_recommendation(clause: str, analysis: str) -> str:
    prompt = (
        f"Given this legal clause:\n{clause}\n\n"
        f"And its analysis:\n{analysis}\n\n"
        "Provide a very short, actionable 'Suggested Fix' or 'Recommendation' "
        "(1-2 sentences max). Return only the recommendation text."
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "Unable to generate recommendation at this time."


def ask_contract_question(question: str, clauses: List[dict]) -> str:
    context = "\n\n".join(
        [f"Clause: {c.get('clause', '')}\nAnalysis: {c.get('analysis', '')}" for c in clauses]
    )
    prompt = (
        "You are a helpful legal AI assistant. Based on the following analyzed "
        "contract clauses, answer the user's question clearly and concisely.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't process that question. Error: {str(e)}"


def build_report_text(clauses: List[dict], total: int, high: int, medium: int, low: int, overall: str, risk_score: int) -> str:
    report = "CONTRACT RISK ANALYSIS REPORT\n" + "=" * 60 + "\n\n"
    report += f"Total Clauses: {total}  |  High: {high}  |  Medium: {medium}  |  Low: {low}\n"
    report += f"Overall Risk: {overall}  |  Risk Score: {risk_score}%\n" + "=" * 60 + "\n\n"
    for i, r in enumerate(clauses):
        report += f"[{i+1}] {r['risk_level']} RISK\n" + "-" * 40 + "\n"
        report += f"Clause: {r['clause']}\n\nAnalysis: {r['analysis']}\n\n"
        report += f"Recommendation: {r.get('recommendation', '')}\n\n"
    return report


# ================================================================
# MODELS
# ================================================================
class ClauseItem(BaseModel):
    clause: str
    analysis: str
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    clauses: List[ClauseItem]


# ================================================================
# ROUTES
# ================================================================
@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "ClauseGuard / ContractRisk AI API",
        "endpoints": ["/api/analyze", "/api/chat", "/health"]
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_contract(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    if os.getenv("GROQ_API_KEY") == "gsk_placeholder_please_set_in_env":
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not configured. Please set your GROQ_API_KEY in the .env file."
        )

    contract_text = ""
    filename = "Pasted Contract Text"

    if file:
        filename = file.filename or "Uploaded Contract.pdf"
        try:
            content = await file.read()
            reader = PdfReader(io.BytesIO(content))
            extracted_pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            contract_text = "\n".join(extracted_pages).strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF file: {str(e)}")
    elif text and len(text.strip()) > 50:
        contract_text = text.strip()

    if not contract_text or len(contract_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF or paste contract text (minimum 50 characters required)."
        )

    try:
        # Build and invoke LangGraph agent
        agent = build_agent()
        result = agent.invoke({"text": contract_text})

        raw_results = result.get("results", [])
        clauses = []

        # Enrich with recommendations and standardized risk level
        for r in raw_results:
            clause_content = r.get("clause", "")
            analysis_content = r.get("analysis", "")
            risk = classify_risk(analysis_content)
            rec = generate_recommendation(clause_content, analysis_content)

            clauses.append({
                "clause": clause_content,
                "analysis": analysis_content,
                "risk_level": risk,
                "recommendation": rec
            })

        # Calculate metrics (1:1 with app.py)
        high = sum(1 for c in clauses if c["risk_level"] == "High")
        medium = sum(1 for c in clauses if c["risk_level"] == "Medium")
        low = sum(1 for c in clauses if c["risk_level"] == "Low")
        total = len(clauses)
        overall = "High" if high > 0 else ("Medium" if medium > 0 else "Low")
        risk_score = max(0, min(100, int(((high * 3 + medium * 1.5) / (total * 3)) * 100))) if total > 0 else 0

        # Generate downloadable report
        report_text = build_report_text(clauses, total, high, medium, low, overall, risk_score)

        return {
            "success": True,
            "filename": filename,
            "total": total,
            "high": high,
            "medium": medium,
            "low": low,
            "overall": overall,
            "risk_score": risk_score,
            "clauses": clauses,
            "report_text": report_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")


@app.post("/api/chat")
def chat_with_contract(payload: ChatRequest):
    if os.getenv("GROQ_API_KEY") == "gsk_placeholder_please_set_in_env":
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not configured. Please set your GROQ_API_KEY in the .env file."
        )

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    clauses_dicts = [c.model_dump() for c in payload.clauses]
    answer = ask_contract_question(payload.question, clauses_dicts)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
