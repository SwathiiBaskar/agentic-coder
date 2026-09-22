#response fetching, streaming response
import asyncio
from typing import Any, AsyncGenerator
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from client.response import EventType, StreamEvent, TextDelta, TokenUsage
load_dotenv()

class LLMClient:
    def __init__(self) -> None:
        self._client:AsyncOpenAI|None=None
        self._max_retries:int=3

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key=os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client=AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client
    
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client=None
    async def chat_completion(self, messages: list[dict[str, Any]], stream: bool=True)-> AsyncGenerator[StreamEvent, None]:
        #llms are stateless
        client=self._get_client()
        kwargs={
            "model":"nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": messages,
            "stream":stream
        }
        if stream:
            async for event in self._stream_response(client, kwargs):
                yield event
        else:
            event=await self._non_stream_response(client, kwargs)
            yield event
        return
            
    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response=await client.chat.completions.create(**kwargs)
        usage: TokenUsage|None=None
        finish_reason:str|None=None
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage=TokenUsage(prompt_tokens=chunk.usage.prompt_tokens,
                                    completion_tokens=chunk.usage.completion_tokens,
                                    total_tokens=chunk.usage.total_tokens,
                                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens
                                )
            if not chunk.choices:
                continue
            choice=chunk.choices[0]
            delta=choice.delta
            if choice.finish_reason:
                finish_reason=choice.finish_reason
            if delta.content:
                yield StreamEvent(
                    type=EventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)
                )
        yield StreamEvent(
            EventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )
    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        response=await client.chat.completions.create(**kwargs)
        choice=response.choices[0]
        message=choice.message
        text_delta=None
        if message.content:
            text_delta=TextDelta(content=message.content)
        usage=None
        if response.usage:
            usage=TokenUsage(prompt_tokens=response.usage.prompt_tokens,
                             completion_tokens=response.usage.completion_tokens,
                             total_tokens=response.usage.total_tokens,
                             cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                            )
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage
        )
            