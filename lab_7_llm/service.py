"""
Web service for model inference.
"""
import json
import os
from pathlib import Path

import pandas as pd
# pylint: disable=undefined-variable
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from config.constants import PROJECT_ROOT
from lab_7_llm.main import LLMPipeline, TaskDataset


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    server = FastAPI()

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r", encoding='utf-8') as settings_path:
        settings = json.load(settings_path)
    task_ds = TaskDataset(pd.DataFrame())
    llm_infer = LLMPipeline(model_name=settings["parameters"]["model"], dataset=task_ds,
                            max_length=120, batch_size=1, device='cpu')

    return server, llm_infer


app, pipeline = init_application()
app_dir = os.path.dirname(__file__)
assets_abs_file_path = os.path.join(app_dir, "assets")
app.mount("/assets", StaticFiles(directory=assets_abs_file_path), name="assets")

jinja_template = Jinja2Templates(directory=str(Path(app_dir, 'assets')))


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
    sample = question.question.split('|')
    try:
        sample_tuple = (sample[0], sample[1])
    except IndexError:
        sample_tuple = (sample[0], sample[0])
    return {
        'infer': pipeline.infer_sample(sample_tuple)
    }


if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8000, reload=True)
