import os
import httpx
import hashlib
import logging
from openai import OpenAI

_global_endpoints = {}


def _get_endpoint(base_url, api_key):
    url_hash = hashlib.md5(base_url.encode()).hexdigest()

    global _global_endpoints
    if url_hash not in _global_endpoints:
        _global_endpoints[url_hash] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(
                verify=False  # important for company use
            )
        )

    return _global_endpoints[url_hash]


def get_model_response(model, messages, **kwargs):
    endpoint = _get_endpoint(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    resp = endpoint.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return (resp.choices[0].message.content.strip(),
            resp.choices[0].finish_reason)


def get_embedding(text, model):
    endpoint = _get_endpoint(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    resp = endpoint.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return resp.data[0].embedding
