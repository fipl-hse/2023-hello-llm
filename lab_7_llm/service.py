"""
Web service for model inference.
"""
# pylint: disable=undefined-variable
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from config.constants import PROJECT_ROOT
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@dataclass
class Query:
    """
    Abstraction which contains text of the query.
    """
    premise: str
    hypothesis: str


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    server = FastAPI()
    server.mount("/assets", StaticFiles(directory="assets"), name="assets")

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        configs = json.load(settings_file)

    data_loader = RawDataImporter(configs['parameters']['dataset'])
    data_loader.obtain()
    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    llm = LLMPipeline(configs['parameters']['model'], dataset, 120, 64, 'cpu')
    return server, llm


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
    templates = Jinja2Templates(directory='assets')
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer a query from webservice.

    Args:
        query (Query): user's query.

    Returns:
        dict: a dictionary with a prediction.
    """
    sample = (query.premise, query.hypothesis)
    labels_mapping = pipeline.get_config()['id2label']
    prediction = pipeline.infer_sample(sample)
    return {'infer': labels_mapping.get(prediction)}
