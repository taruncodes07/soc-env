from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from server.environment import Environment
from server.models import Action, Observation, EpisodeState
from fastapi.responses import HTMLResponse

app = FastAPI(title="SOC-Env", description="OpenEnv compliant SOC environment", version="1.0.0")

env_instance = Environment()

episode_rewards = []

class StepResponse(BaseModel):
    observation: Observation
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def read_root():
    return """
    <html>
        <head>
            <title>SOC-Env Status</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #121212; color: #ffffff; }
                .container { text-align: center; background: #1e1e1e; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
                h1 { color: #4caf50; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛡️ SOC-Env Serving</h1>
                <p>Environment is active and ready for inference.</p>
                <p><a href="/docs" style="color: #64b5f6;">View API Documentation</a></p>
                <p><a href="/metrics" style="color: #64b5f6;">View Metrics</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/metrics")
def get_metrics():
    return {"episode_rewards": episode_rewards}

@app.post("/reset", response_model=Observation)
def reset(
    task: str = Query("task_1", description="The task ID to load"),
    randomize: bool = Query(True, description="Whether to randomize the scenario"),
    seed: Optional[int] = Query(None, description="Random seed for deterministic reset")
):
    try:
        if seed is not None:
            import random
            random.seed(seed)
        obs = env_instance.reset(task, randomize=randomize)
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
        if done:
            episode_rewards.append(env_instance.state.cumulative_reward)
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

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
