"""
FastAPI listing.

1. Install dependencies:
    - FastAPI: pip install fastapi[all]
    - uvicorn: pip install uvicorn
2. Initialize FastAPI instance
3. (Optionally) Mount static folder
4. Define endpoints
5. Run local server: uvicorn seminars.seminar_02_06_2024.try_fastapi:app --reload
"""
import random

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')

# 2. Initialize FastAPI instance
app = FastAPI()

# 3. (Optionally) Mount static folder
app.mount(
    '/static',
    StaticFiles(directory='seminars/seminar_02_06_2024/static'),
    name='static'
)


# 4. Define endpoints
@app.get('/')
async def handle_root_endpoint() -> dict:
    """
    Root endpoint of application.
    """
    return {'response': 'Hello, LLM!'}


@app.get('/templates', response_class=HTMLResponse)
async def handle_get_request(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when no dynamic data is loaded.
    """
    templates = Jinja2Templates(directory='seminars/seminar_02_06_2024/templates')
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/templates_with_static', response_class=HTMLResponse)
async def handle_get_with_static_request(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when dynamic data is loaded.
    """
    templates = Jinja2Templates(directory='seminars/seminar_02_06_2024/templates')
    return templates.TemplateResponse(
        'index_with_static.html',
        {
            'request': request,
            'random_name': random.choice(('Alice', 'Bob', 'Tom', 'John'))
        }
    )
