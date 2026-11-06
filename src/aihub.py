#!/usr/bin/env python3
"""
Description: Minimal OpenAI SDK transport for Stanford AI Hub Azure deployments.
"""

# Imports
import json
import httpx
import openai

# Constants
GATEWAY = "https://aihubapi.stanfordhealthcare.org/azure-openai"


class HubTransport(httpx.BaseTransport):
    def __init__(self, api_key):
        self.api_key = api_key
        self.transport = httpx.HTTPTransport()

    def handle_request(self, request):
        payload = json.loads(request.content)
        model = payload.get("model", "chat")
        resource = request.url.path.removeprefix("/azure-openai")
        url = request.url.copy_with(
            path=f"/azure-openai/deployments/{model}{resource}",
            params=dict(request.url.params) | {"api-version": "2025-04-01-preview"},
        )
        headers = dict(request.headers)
        for name in ("authorization", "x-api-key", "x-goog-api-key", "anthropic-version"):
            headers.pop(name, None)
        headers["api-key"] = self.api_key
        forwarded = httpx.Request(request.method, url, headers=headers,
                                  content=request.content, extensions=request.extensions)
        return self.transport.handle_request(forwarded)

    def close(self):
        self.transport.close()


def get_openai_client(api_key, gateway_url="https://aihubapi.stanfordhealthcare.org"):
    if gateway_url.rstrip("/") != "https://aihubapi.stanfordhealthcare.org":
        raise ValueError("only Stanford AI Hub is supported")
    return openai.OpenAI(
        api_key="aihub", base_url=GATEWAY,
        http_client=httpx.Client(transport=HubTransport(api_key)),
    )
