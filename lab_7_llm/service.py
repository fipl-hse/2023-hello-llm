"""
Web service for model inference.
"""
import json
import os
from pathlib import Path

# pylint: disable=undefined-variable
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.constants import PROJECT_ROOT
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    server = FastAPI()

    with open(PROJECT_ROOT/'lab_7_llm'/'settings.json', "r", encoding='utf-8') as settings_path:
        settings = json.load(settings_path)
    ds_obtained = RawDataImporter(settings["parameters"]["dataset"])
    ds_obtained.obtain()
    preprocessed_ds = RawDataPreprocessor(ds_obtained.raw_data)
    preprocessed_ds.transform()
    task_ds = TaskDataset(preprocessed_ds.data.head(100))
    llm_infer = LLMPipeline(model_name=settings["parameters"]["model"], dataset=task_ds,
                            max_length=120, batch_size=64, device='cpu')

    return server, llm_infer


init_server = init_application()

app, pipeline = init_server[0], init_server[1]
app_dir = os.path.dirname(__file__)
assets_abs_file_path = os.path.join(app_dir, "assets")
print(str(assets_abs_file_path))
app.mount("/assets", StaticFiles(directory=assets_abs_file_path), name="assets")

BASE_DIR = Path(__file__).resolve().parent
jinja_template = Jinja2Templates(directory=str(Path(BASE_DIR, 'assets')))


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


if __name__ == "__main__":
    uvicorn.run("service:app", host='127.0.0.1', port=8000, reload=True)
