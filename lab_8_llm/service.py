"""
Web service for model inference.
"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pandas import DataFrame
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

    application = FastAPI()
    application.mount(
        "/assets",
        StaticFiles(directory=PROJECT_ROOT / "lab_8_llm" / "assets"),
        name="assets"
    )

    root_path = Path(__file__).parent
    settings_path = LabSettings(root_path / "settings.json")

    dataset = TaskDataset(DataFrame())
    llm_pipeline = LLMPipeline(model_name=settings_path.parameters.model,
                               dataset=dataset,
                               max_length=120,
                               batch_size=1,
                               device="cpu")
    return application, llm_pipeline


app, pipeline = init_application()


@dataclass
class Query:
    """
    Abstraction class which contains text of the query.
    """
    question: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint of application.
    """

    templates = Jinja2Templates(directory="assets")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer endpoint of application.
    """

    prediction = pipeline.infer_sample(query.question)
    return {'infer': prediction}


if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8000, reload=True)
