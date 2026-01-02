import time
import uuid
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator


class CompletionRequest(BaseModel):
    model: str = "glm-4.7-fp8"
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    user: Optional[str] = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        if isinstance(v, str):
            if len(v) == 0:
                raise ValueError("prompt cannot be empty")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("prompt list cannot be empty")
            for p in v:
                if not isinstance(p, str) or len(p) == 0:
                    raise ValueError("each prompt must be a non-empty string")
        return v


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="The role of the message author")
    content: str = Field(..., description="The message content")


class ChatCompletionRequest(BaseModel):
    model: str = "glm-4.7-fp8"
    messages: List[ChatMessage] = Field(..., description="The messages to generate chat completions for")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    user: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if len(v) == 0:
            raise ValueError("messages cannot be empty")
        return v


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length", "error"]] = None
    logprobs: Optional[dict] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "glm-4.7-fp8"
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "error"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "glm-4.7-fp8"
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[Literal["stop", "length", "error"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = "glm-4.7-fp8"
    choices: List[StreamChoice]


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = "glm-4.7-fp8"
    choices: List[CompletionChoice]


class ErrorResponse(BaseModel):
    error: dict = Field(..., description="Error details")


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    engine_loaded: bool
    worker_running: bool
