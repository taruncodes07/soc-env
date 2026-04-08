from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from server.environment import Environment
from server.models import Action, Observation, EpisodeState

app = FastAPI(title="SOC-Env", description="OpenEnv compliant SOC environment", version="1.0.0")

env_instance = Environment()

from fastapi.responses import RedirectResponse

class StepResponse(BaseModel):
    observation: Observation
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/reset", response_model=Observation)
def reset(task: str = Query("task_1", description="The task ID to load")):
    try:
        obs = env_instance.reset(task)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    if env_instance.state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    if env_instance.state.done:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset.")
        
    try:
        obs, reward, done, info = env_instance.step(action)
        return StepResponse(
            observation=obs,
            reward=reward.model_dump(),
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=EpisodeState)
def state():
    if env_instance.state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    return env_instance.state

def start():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
