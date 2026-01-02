import asyncio
import json
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Union, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import ValidationError

from src.server.config import ServerConfig
from src.server.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    CompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatMessage,
    StreamChoice,
    StreamDelta,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)
from src.server.engine_worker import EngineWorker, InferenceRequest, InferenceResult
from src.server.tokenizer import Tokenizer, MockTokenizer
from src.server.metrics import metrics
from src.server.logging_config import setup_logging, RequestLogger


class AppState:
    def __init__(self):
        self.config: Optional[ServerConfig] = None
        self.worker: Optional[EngineWorker] = None
        self.tokenizer: Optional[Union[Tokenizer, MockTokenizer]] = None
        self.logger = None
        self.shutdown_event = asyncio.Event()


state = AppState()


def create_app(config: Optional[ServerConfig] = None, use_mock: bool = False) -> FastAPI:
    if config is None:
        config = ServerConfig.from_env()
    state.config = config
    state.logger = setup_logging(config.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_event_loop()
        state.worker = EngineWorker(config, use_mock=use_mock)
        if use_mock:
            state.tokenizer = MockTokenizer()
        else:
            state.tokenizer = Tokenizer(config.model_dir)
        state.tokenizer.load()
        try:
            state.worker.start(loop)
            state.logger.info("Engine worker started")
        except Exception as e:
            state.logger.error(f"Failed to start engine worker: {e}")
            raise

        try:
            def handle_shutdown(sig, frame):
                state.logger.info(f"Received signal {sig}, initiating shutdown")
                state.shutdown_event.set()
            signal.signal(signal.SIGTERM, handle_shutdown)
            signal.signal(signal.SIGINT, handle_shutdown)
        except ValueError:
            pass
        yield
        state.logger.info("Shutting down engine worker")
        await state.worker.stop()
        state.logger.info("Engine worker stopped")

    app = FastAPI(
        title="GLM-4.7-FP8 Inference Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz():
        engine_loaded = state.worker is not None and state.worker.is_running()
        worker_running = engine_loaded
        if engine_loaded and worker_running:
            return ReadyResponse(status="ok", engine_loaded=True, worker_running=True)
        raise HTTPException(
            status_code=503,
            detail=ReadyResponse(
                status="not_ready",
                engine_loaded=engine_loaded,
                worker_running=worker_running,
            ).model_dump(),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        return metrics.to_prometheus()

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, raw_request: Request):
        request_id = str(uuid.uuid4())
        req_logger = RequestLogger(state.logger, request_id)
        if state.worker is None or not state.worker.is_running():
            raise HTTPException(status_code=503, detail="Engine not ready")
        if state.worker.get_queue_depth() >= state.config.queue_size:
            raise HTTPException(status_code=429, detail="Queue full, try again later")
        if state.worker.get_active_count() >= state.config.max_concurrency:
            raise HTTPException(status_code=429, detail="Max concurrency reached")
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        if len(prompts) > 1:
            raise HTTPException(status_code=400, detail="Batch prompts not supported yet")
        prompt = prompts[0]
        token_ids = state.tokenizer.encode(prompt)
        if len(token_ids) > state.config.max_prompt_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long: {len(token_ids)} > {state.config.max_prompt_tokens}",
            )
        max_tokens = min(request.max_tokens, state.config.max_new_tokens)
        req_logger.set_prompt_tokens(len(token_ids))
        inference_request = InferenceRequest(
            request_id=request_id,
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stream=request.stream,
        )
        try:
            if request.stream:
                return StreamingResponse(
                    _stream_completion(inference_request, request.model, req_logger),
                    media_type="text/event-stream",
                )
            else:
                result = await state.worker.submit_request(inference_request)
                output_text = state.tokenizer.decode(result.tokens)
                req_logger.gen_tokens = result.completion_tokens
                req_logger.complete(result.finish_reason)
                return CompletionResponse(
                    id=f"cmpl-{request_id[:24]}",
                    model=request.model,
                    choices=[
                        CompletionChoice(
                            index=0,
                            text=output_text,
                            finish_reason=result.finish_reason,
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens,
                    ),
                )
        except RuntimeError as e:
            req_logger.error(str(e))
            metrics.inc_error()
            raise HTTPException(status_code=503, detail=str(e))

    async def _stream_completion(
        request: InferenceRequest, model: str, req_logger: RequestLogger
    ) -> AsyncIterator[str]:
        response_id = f"cmpl-{request.request_id[:24]}"
        created = int(time.time())
        try:
            token_stream = await state.worker.submit_request(request)
            async for token in token_stream:
                req_logger.token_generated()
                text = state.tokenizer.decode_single(token)
                chunk = CompletionChunk(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[
                        CompletionChoice(index=0, text=text, finish_reason=None)
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            final_chunk = CompletionChunk(
                id=response_id,
                created=created,
                model=model,
                choices=[
                    CompletionChoice(index=0, text="", finish_reason="stop")
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            req_logger.complete("stop")
        except Exception as e:
            req_logger.error(str(e))
            metrics.inc_error()
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        request_id = str(uuid.uuid4())
        req_logger = RequestLogger(state.logger, request_id)
        if state.worker is None or not state.worker.is_running():
            raise HTTPException(status_code=503, detail="Engine not ready")
        if state.worker.get_queue_depth() >= state.config.queue_size:
            raise HTTPException(status_code=429, detail="Queue full, try again later")
        if state.worker.get_active_count() >= state.config.max_concurrency:
            raise HTTPException(status_code=429, detail="Max concurrency reached")
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = state.tokenizer.apply_chat_template(messages_dict)
        token_ids = state.tokenizer.encode(prompt)
        if len(token_ids) > state.config.max_prompt_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long: {len(token_ids)} > {state.config.max_prompt_tokens}",
            )
        max_tokens = min(request.max_tokens, state.config.max_new_tokens)
        req_logger.set_prompt_tokens(len(token_ids))
        inference_request = InferenceRequest(
            request_id=request_id,
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stream=request.stream,
        )
        try:
            if request.stream:
                return StreamingResponse(
                    _stream_chat_completion(inference_request, request.model, req_logger),
                    media_type="text/event-stream",
                )
            else:
                result = await state.worker.submit_request(inference_request)
                output_text = state.tokenizer.decode(result.tokens)
                req_logger.gen_tokens = result.completion_tokens
                req_logger.complete(result.finish_reason)
                return ChatCompletionResponse(
                    id=f"chatcmpl-{request_id[:24]}",
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=output_text),
                            finish_reason=result.finish_reason,
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens,
                    ),
                )
        except RuntimeError as e:
            req_logger.error(str(e))
            metrics.inc_error()
            raise HTTPException(status_code=503, detail=str(e))

    async def _stream_chat_completion(
        request: InferenceRequest, model: str, req_logger: RequestLogger
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{request.request_id[:24]}"
        created = int(time.time())
        first_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(index=0, delta=StreamDelta(role="assistant"), finish_reason=None)
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"
        try:
            token_stream = await state.worker.submit_request(request)
            async for token in token_stream:
                req_logger.token_generated()
                text = state.tokenizer.decode_single(token)
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(index=0, delta=StreamDelta(content=text), finish_reason=None)
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            final_chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=model,
                choices=[
                    StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            req_logger.complete("stop")
        except Exception as e:
            req_logger.error(str(e))
            metrics.inc_error()
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return app
