"""
Web service for model inference.
"""
# pylint: disable=undefined-variable, duplicate-code
import os
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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

    settings = LabSettings(PROJECT_ROOT / "lab_8_llm" / 'settings.json')
    task_ds = TaskDataset(pd.DataFrame())
    llm_infer = LLMPipeline(model_name=settings.parameters.model, dataset=task_ds,
                            max_length=120, batch_size=1, device='cpu')

    return server, llm_infer


app, pipeline = init_application()

app.mount("/assets", StaticFiles(directory=PROJECT_ROOT / "lab_8_llm" / 'assets'), name="assets")

jinja_template = Jinja2Templates(directory=PROJECT_ROOT / "lab_8_llm" / 'assets')


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """

    Args:
        request: Client's request to the server
    Returns:
        HTMLResponse: Main instance of the app's frontend part
    """
    return jinja_template.TemplateResponse(
        name="index.html",
        request=request
    )


@dataclass
class Query(BaseModel):
    """
    Class that initializes the question to infer, gets it from fetchAPI
    """
    question: str


@app.post('/infer')
async def infer(question: Query) -> dict:
    """

    Args:
        question: Query class instance with string to infer

    Returns:
        dict: dict with the results of sample inference
    """
    return {
        'infer': pipeline.infer_sample((question.question,))
    }
