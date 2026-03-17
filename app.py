from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import build_agent
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Healthcare Agent API")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory agent cache mapping session IDs to agent executors
session_agents = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class StepDetail(BaseModel):
    tool: str
    tool_input: str
    log: str
    observation: str

class ChatResponse(BaseModel):
    output: str
    intermediate_steps: List[StepDetail] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if request.session_id not in session_agents:
        session_agents[request.session_id] = build_agent(verbose=True)
        
    agent = session_agents[request.session_id]
    
    try:
        result = agent.invoke({"input": request.message})
        
        steps = []
        # Serialize the trace (AgentAction, Observation pairs)
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                steps.append(StepDetail(
                    tool=getattr(action, "tool", "Unknown Tool"),
                    tool_input=str(getattr(action, "tool_input", "")),
                    log=getattr(action, "log", ""),
                    observation=str(observation)
                ))
                
        return ChatResponse(
            output=result.get("output", "No response generated."),
            intermediate_steps=steps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
