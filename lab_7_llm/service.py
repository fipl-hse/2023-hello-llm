"""
Web service for model inference.
"""
# pylint: disable=undefined-variable
import json

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

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
