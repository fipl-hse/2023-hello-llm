"""
Web service for model inference.
"""
import json

# pylint: disable=undefined-variable
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

    path = PROJECT_ROOT / 'lab_7_llm' / 'assets'

    server.mount("/assets", StaticFiles(directory=str(path)), name='assets')

    task_dataset = TaskDataset(pd.DataFrame())

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as config:
        data = json.load(config)

    pipe = LLMPipeline(data['parameters']['model'],
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
    templates = Jinja2Templates(directory='assets')
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer the query in web-site
    """
    sample = query.question.split("|")

    label_mapping = {"0": "entailment",
                     "1": "contradiction",
                     "2": "neutral"}

    prediction = pipeline.infer_sample(sample + sample if '|' not in sample else sample)

    return {"infer": label_mapping.get(prediction)}

if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8000, reload=True)
