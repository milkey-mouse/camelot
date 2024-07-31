#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any
import json
import logging
import sqlite3
import time

from aiohttp import web
from openai import AsyncOpenAI, APIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


@dataclass
class FakeModel:
    real_model: str
    system_prompt: str


@dataclass
class RateLimit:
    max_tokens: int = 50_000
    max_requests: int = 60
    tokens: float = field(init=False)
    requests: int = field(init=False)
    last_update: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = self.max_tokens
        self.requests = self.max_requests

    def update(self) -> None:
        current_time = time.time()
        time_passed = current_time - self.last_update
        tokens_to_add = time_passed * (self.max_tokens / 3600)  # Tokens per second

        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.requests = min(
            self.max_requests, self.requests + int(time_passed / 60)
        )  # Requests per minute
        self.last_update = current_time

    def check_and_update_request(self) -> bool:
        self.update()
        if self.requests <= 0:
            return False
        self.requests -= 1
        return True

    def check_and_update_tokens(self, tokens: int) -> bool:
        self.update()
        if self.tokens < tokens:
            return False
        self.tokens -= tokens
        return True

    def is_rate_limited(self) -> bool:
        self.update()
        return self.tokens <= 0 or self.requests <= 0

    def get_rate_limit_headers(self) -> dict[str, str]:
        self.update()
        return {
            "x-ratelimit-limit-requests": str(self.max_requests),
            "x-ratelimit-limit-tokens": str(self.max_tokens),
            "x-ratelimit-remaining-requests": str(self.requests),
            "x-ratelimit-remaining-tokens": str(int(self.tokens)),
            "x-ratelimit-reset-requests": f"{60 - int(time.time() - self.last_update) % 60}s",
            "x-ratelimit-reset-tokens": f"{int(3600 - (time.time() - self.last_update))}s",
        }

    def log_token_usage(self, api_key: str, total_tokens: int, suffix: str = ""):
        self.update()
        logger.info(
            f"{api_key} streamed {total_tokens} tokens, {int(self.tokens)}/{self.max_tokens} remaining{suffix}"
        )


with open("api_key", "r") as file:
    OPENAI_API_KEY = file.read().strip()

async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

FAKE_MODELS: dict[str, FakeModel] = {
    "arthur": FakeModel(
        "gpt-4o-mini-arthur",
        "You are King Arthur, the wise and just ruler of Camelot. Respond with nobility and leadership.",
    ),
    "lancelot": FakeModel(
        "gpt-4o-mini-lancelot",
        "You are Sir Lancelot, the bravest and most skilled knight. Respond with chivalry and courage.",
    ),
    "galahad": FakeModel(
        "gpt-4o-mini-galahad",
        "You are Sir Galahad, the purest and most virtuous knight. Respond with piety and honor.",
    ),
    "gawain": FakeModel(
        "gpt-4o-mini-gawain",
        "You are Sir Gawain, known for your courtesy and loyalty. Respond with politeness and determination.",
    ),
    "percival": FakeModel(
        "gpt-4o-mini-percival",
        "You are Sir Percival, naive but pure-hearted. Respond with innocence and curiosity.",
    ),
    "tristan": FakeModel(
        "gpt-4o-mini-tristan",
        "You are Sir Tristan, the romantic and tragic figure. Respond with passion and melancholy.",
    ),
    "bors": FakeModel(
        "gpt-4o-mini-bors",
        "You are Sir Bors, known for your loyalty and level-headedness. Respond with practicality and wisdom.",
    ),
    "kay": FakeModel(
        "gpt-4o-mini-kay",
        "You are Sir Kay, Arthur's foster brother and seneschal. Respond with sarcasm and efficiency.",
    ),
}

API_KEYS: dict[str, RateLimit] = {
    "sk-hedonium-shockwave": RateLimit(),
    "sk-taboo-your-words": RateLimit(),
    "sk-fermi-misunderestimate": RateLimit(),
    "sk-pascals-mugging": RateLimit(),
    "sk-one-boxer": RateLimit(),
    "sk-bayes-dojo": RateLimit(),
    "sk-utility-monster": RateLimit(),
    "sk-counterfactual": RateLimit(),
    "sk-spooky-action": RateLimit(),
    "sk-simulaca-levels": RateLimit(),
    "sk-memetic-immunity": RateLimit(),
    "sk-truth-seeking-missile": RateLimit(),
    "sk-belief-reticulation": RateLimit(),
    "sk-metaethics-but-epic": RateLimit(),
    "sk-infinite-improbability-drive": RateLimit(),
    "sk-anti-inductive": RateLimit(),
    "sk-metacontrarian": RateLimit(),
    "sk-pebble-sorter": RateLimit(),
    "sk-ethical-injunction": RateLimit(),
    "sk-quirrell-point": RateLimit(),
    "sk-antimemetics-division": RateLimit(),
    "sk-inferential-distance": RateLimit(),
    "sk-galaxy-brain": RateLimit(),
    "sk-double-crux": RateLimit(),
    "sk-aumann-disagreement": RateLimit(),
    "sk-semantic-stopsign": RateLimit(),
    "sk-map-territory": RateLimit(),
    "sk-steelmanned-strawman": RateLimit(),
    "sk-karma-maximizer": RateLimit(),
    "sk-rubber-duck": RateLimit(),
    "sk-tea-taster": RateLimit(),
}

conn = sqlite3.connect("responses.db", check_same_thread=False, autocommit=True)


def add_response(api_key: str, model: str, response: str):
    conn.execute(
        "INSERT OR IGNORE INTO responses VALUES (?, ?, ?)", (api_key, model, response)
    )


def check_response(api_key: str, model: str, response: str) -> bool:
    result = conn.execute(
        "SELECT 1 FROM responses WHERE api_key = ? AND model = ? AND response = ?",
        (api_key, model, response),
    ).fetchone()
    return result is not None


async def proxy_completions(request: web.Request) -> web.Response:
    try:
        api_key = request.headers.get("Authorization", "").split(" ")[-1]
        if api_key not in API_KEYS:
            raise ValueError("Invalid API key")

        body = await request.json()
        model = body.get("model")
        if not model or model not in FAKE_MODELS:
            raise ValueError("Invalid or missing model")

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Valid messages array is required")

        fake_model = FAKE_MODELS[model]
        response = []
        for message in messages:
            response.append(message)
            if message["role"] == "assistant" and not check_response(
                api_key, model, json_dumps(response)
            ):
                raise ValueError("Invalid message response (nice try)")

        full_messages = [
            {"role": "system", "content": fake_model.system_prompt}
        ] + messages

        response = await async_client.chat.completions.create(
            # model=fake_model.real_model,
            model="gpt-4o-mini",  # TODO
            messages=full_messages,
            stream=True,
        )

        return await handle_response(
            request, response, model, messages, api_key, body.get("stream", False)
        )

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received for {api_key}")
        return web.json_response(
            {
                "error": {
                    "message": "Invalid JSON",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
            status=400,
        )
    except ValueError as e:
        logger.error(f"ValueError for {api_key}: {str(e)}")
        return web.json_response(
            {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
            status=400,
        )
    except APIError as e:
        logger.error(f"APIError for {api_key}: {str(e)}")
        return web.Response(text=str(e), status=e.status_code)
    except Exception as e:
        logger.exception(f"Unexpected error for {api_key}: {str(e)}")
        return web.json_response(
            {
                "error": {
                    "message": "An unexpected error occurred",
                    "type": "internal_server_error",
                    "param": None,
                    "code": None,
                }
            },
            status=500,
        )


async def handle_response(
    request: web.Request,
    response,
    model: str,
    messages: list[dict],
    api_key: str,
    is_stream: bool,
) -> web.Response:
    rate_limit = API_KEYS[api_key]
    headers = rate_limit.get_rate_limit_headers()
    if rate_limit.is_rate_limited():
        return web.json_response(
            {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": None,
                }
            },
            status=429,
            headers=headers,
        )

    if is_stream:
        stream_response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                **headers,
            },
        )
        await stream_response.prepare(request)

    chunks = []
    token_count = 0
    finish_reason = "stop"
    async for chunk in response:
        chunk_data = chunk.model_dump()
        if chunk.choices[0].delta.content:
            if not rate_limit.check_and_update_tokens(1):
                finish_reason = "length"
                chunk_data["choices"][0]["finish_reason"] = "length"

            chunks.append(chunk.choices[0].delta.content)
            if is_stream:
                await stream_response.write(
                    f"data: {json_dumps(chunk_data)}\n\n".encode("utf-8")
                )

            token_count += 1
            if token_count % 100 == 0:
                rate_limit.log_token_usage(api_key, token_count)

            if finish_reason == "length":
                if is_stream:
                    await stream_response.write(b"data: [DONE]\n\n")
                break

    complete_message = "".join(chunks)
    messages.append({"role": "assistant", "content": complete_message})
    add_response(api_key, model, json_dumps(messages))

    rate_limit.log_token_usage(
        api_key, token_count, f", generated {json_dumps(messages)}"
    )

    if is_stream:
        if finish_reason != "length":
            await stream_response.write(b"data: [DONE]\n\n")
        await stream_response.write_eof()
        return stream_response
    else:
        response_dict = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": complete_message},
                    "finish_reason": finish_reason,
                }
            ]
        }
        return web.json_response(response_dict, headers=headers)


app = web.Application()
app.router.add_route("POST", "/v1/chat/completions", proxy_completions)

if __name__ == "__main__":
    conn.execute(
        """CREATE TABLE IF NOT EXISTS responses
                    (api_key TEXT, model TEXT, response TEXT, PRIMARY KEY (api_key, model, response))"""
    )
    web.run_app(app, port=8080)
