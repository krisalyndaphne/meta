from fastapi import FastAPI
from fastapi import Body
from pydantic import BaseModel

from invoice_audit_env.env import InvoiceAuditEnv
from invoice_audit_env.models import Action

app = FastAPI(title="openenv-invoice-audit")
env = InvoiceAuditEnv()


class ResetRequest(BaseModel):
    task_id: str
    seed: int | None = None


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest | None = Body(default=None)) -> dict:
    task_id = payload.task_id if payload is not None else "easy_single_mismatch"
    seed = payload.seed if payload is not None else None
    obs = env.reset(task_id=task_id, seed=seed)
    return obs.model_dump()


@app.post("/step")
def step(action: Action) -> dict:
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return env.state()
