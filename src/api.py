import uuid
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST
from .model import Model, GeneratorArgs


class API:
    def __init__(self, model: Model):
        self._model = model

        router = APIRouter()
        router.add_api_route('/', self._ping, methods=['POST', 'GET'])
        router.add_api_route('/generate', self._generate, methods=['POST', 'GET'])

        self._app = FastAPI()
        self._app.include_router(router)
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    def run(self, host: str = '0.0.0.0', port: int = 5000):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port)

    async def _ping(self) -> Response:
        return Response(status_code=200)

    async def _generate(self, request: Request) -> StreamingResponse:
        if request.method == 'GET':
            params = request.query_params
        elif request.method == 'POST':
            params = await request.json()

        prompt = params.get('prompt')
        args = GeneratorArgs(
            temperature=float(params.get('temperature', GeneratorArgs.temperature)),
            top_k=float(params.get('top_k', GeneratorArgs.top_k)),
            top_p=float(params.get('top_p', GeneratorArgs.top_p)),
            max_tokens=int(params.get('max_tokens', GeneratorArgs.max_tokens))
        )

        if prompt is None:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail='Prompt is required')

        id = uuid.uuid4()
        async def abort():
            await self._model.abort(id)

        task = BackgroundTask(abort)
        async def generate():
            async for data in self._model.generate(prompt, id, args):
                yield data

        return StreamingResponse(generate(), media_type='text/event-stream', background=task)

    async def _reload(self):
        pass
