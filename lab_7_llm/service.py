"""
Web service for model inference.
"""
# pylint: disable=undefined-variable
try:
    from fastapi import FastAPI
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')
    FastAPI = None

from lab_7_llm.main import LLMPipeline


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """


app, pipeline = None, None
