"""Amazon Bedrock Claude helper using the normal AWS credential chain."""

from __future__ import annotations

import json
import os
from functools import lru_cache


BEDROCK_MODEL_NAME_MAP = {
    "claude": "anthropic.claude-v2:1",
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "hiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "opus": "anthropic.claude-3-opus-20240229-v1:0",
}


@lru_cache(maxsize=1)
def _client():
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_DEFAULT_REGION is not set for Amazon Bedrock.")
    return boto3.client("bedrock-runtime", region_name=region)


def get_claude_response(
    llm: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0,
    tools=None,
    tool_choice=None,
) -> str:
    model = BEDROCK_MODEL_NAME_MAP.get(llm, llm if llm.startswith("anthropic.") else None)
    if model is None:
        raise ValueError(f"Unknown LLM alias: {llm}")
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": temperature,
    }
    if tools:
        request_body["tools"] = tools
    if tool_choice:
        request_body["tool_choice"] = tool_choice
    response = _client().invoke_model(body=json.dumps(request_body), modelId=model)
    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]
