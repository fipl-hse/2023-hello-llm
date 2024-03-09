"""
Web service for model inference.
"""
# pylint: disable=undefined-variable, duplicate-code
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_8_llm.main import LLMPipeline, TaskDataset


@dataclass
class Query:
    """
    Abstraction class which contains text of the query.
    """
    question: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    server = FastAPI()
    server.mount(
        "/assets",
        StaticFiles(directory=PROJECT_ROOT / 'lab_8_llm' / 'assets'),
        name="assets"
    )

    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')

    llm_pipeline = LLMPipeline(settings.parameters.model, TaskDataset(pd.DataFrame()), 120, 1, 'cpu')
    return server, llm_pipeline


app, pipeline = init_application()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint.

    Args:
        request (Request): request.

    Returns:
        HTMLResponse: start page of the webservice.
    """
    templates = Jinja2Templates(directory=str(PROJECT_ROOT / 'lab_8_llm' / 'assets'))
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer endpoint of application.
    """
    languages = {"0": "Tatar",
                 "1": "Russian",
                 "2": "Kyrgyz",
                 "3": "Karachay-Balkar",
                 "4": "Bashkir",
                 "5": "Yakut",
                 "6": "Kazakh",
                 "7": "Tuvinian",
                 "8": "Chuvash"}

    prediction = pipeline.infer_sample(query.question)
    return {'infer': languages.get(prediction)}


if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8001, reload=True)
