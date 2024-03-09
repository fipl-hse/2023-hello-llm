"""
Web service for model inference.
"""
# pylint: disable=undefined-variable, duplicate-code
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_8_llm.main import LLMPipeline, TaskDataset


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    server = FastAPI()

    path = PROJECT_ROOT / 'lab_8_llm' / 'assets'

    server.mount("/assets", StaticFiles(directory=str(path)), name='assets')

    task_dataset = TaskDataset(pd.DataFrame())

    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')

    pipe = LLMPipeline(settings.parameters.model,
                       task_dataset,
                       batch_size=1,
                       max_length=120,
                       device='cpu')

    return server, pipe


app, pipeline = init_application()


@dataclass
class Query:
    """
    Abstraction class with question field
    """
    question: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when no dynamic data is loaded.
    """
    templates = Jinja2Templates(directory=PROJECT_ROOT / 'lab_8_llm' / 'assets')
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer the query in web-site
    """

    return {"infer": pipeline.infer_sample(query.question)}

if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8000, reload=True)
