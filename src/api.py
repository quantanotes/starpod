import uuid
from fastapi import APIRouter, FastAPI
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST
from .logger import logger
from .model import Model, GeneratorArgs
from .rwlock import RWLock

class API:
    def __init__(self, model: Model):
        self._model = model

        self._lock = RWLock()

        router = APIRouter()
        router.add_api_route('/', self._ping, methods=['POST', 'GET'])
        router.add_api_route('/generate', self._generate, methods=['POST', 'GET'])
        router.add_api_route('/reset', self._reset, methods=['POST', 'GET'])

        self._app = FastAPI()
        self._app.include_router(router)

    def run(self, host: str = '0.0.0.0', port: int = 5000):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port)

    async def _ping(self) -> Response:
        return Response(status_code=200)

    async def _generate(self, request: Request) -> StreamingResponse:
        if request.method == 'GET':
            params = request.query_params
            prompt = params.get('prompt')
            args = GeneratorArgs(
                temperature=params.get('temperature', GeneratorArgs.temperature),
                top_k=params.get('top_k', GeneratorArgs.top_k),
                top_p=params.get('top_p', GeneratorArgs.top_p),
                max_tokens=params.get('max_tokens', GeneratorArgs.max_tokens)
            )
        elif request.method == 'POST':
            body: dict = await request.json()
            prompt = body.get('prompt')
            args = GeneratorArgs(
                temperature=body.get('temperature', GeneratorArgs.temperature),
                top_k=body.get('top_k', GeneratorArgs.top_k),
                top_p=body.get('top_p', GeneratorArgs.top_p),
                max_tokens=body.get('max_tokens', GeneratorArgs.max_tokens)
            )

        if not prompt:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Prompt is required")

        id = uuid.uuid4()
        async def abort():
            await self._model.abort(id)
        task = BackgroundTask(abort)

        async def generate():
            with self._lock.read_lock:
                async for data in self._model.generate(prompt, id, args):
                    yield data

        return StreamingResponse(generate(), media_type='text/event-stream', background=task)

    def _reset(self):
        with self._lock.write_lock:
            logger.info("Initiating model reset request")
            self._model.reset()
            logger.info("Finished model reset request")