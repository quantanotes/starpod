import uuid
from fastapi import APIRouter, FastAPI
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST
from .model import Model

class API:
    def __init__(self, model: Model):
        self._model = model

        router = APIRouter()
        router.add_api_route('/generate', self._generate, methods=['POST', 'GET'])

        self._app = FastAPI()
        self._app.include_router(router)

    def run(self, host: str = '0.0.0.0', port: int = 5000):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port)

    async def _generate(self, request: Request) -> StreamingResponse:
        if request.method == 'GET':
            prompt = request.query_params.get('prompt')
        elif request.method == 'POST':
            body: dict = await request.json()
            prompt = body.get('prompt')

        if not prompt:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Prompt is required")

        id = uuid.uuid4()

        async def abort():
            await self._model.abort(id)

        _ = BackgroundTask(abort)

        return StreamingResponse(self._model.generate(prompt), media_type='text/event-stream')
