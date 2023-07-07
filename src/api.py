from fastapi import APIRouter, FastAPI
from starlette.responses import StreamingResponse
from model import Model


class API:
    def __init__(self, model: Model):
        self._model = model

        router = APIRouter()
        router.add_api_route('/generate', self._generate, methods=['GET'])

        self._app = FastAPI()
        self._app.include_router(router)

    def run(self, host: str = '0.0.0.0', port: int = 8080):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port)

    async def _generate(self, prompt: str) -> StreamingResponse:
        return StreamingResponse(self._model.generate(prompt), media_type='text/event-stream')
