import asyncio
import base64
import json
from typing import Any, Dict, Optional

import httpx

from app.config import settings


class LLMServiceError(Exception):
    pass


class MedicalLLMService:
    def __init__(self) -> None:
        self.provider = _normalize_provider(settings.LLM_PROVIDER)
        self.api_key = settings.LLM_API_KEY
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.timeout_seconds = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max(settings.LLM_MAX_RETRIES, 0)
        self.vision_model = settings.LLM_VISION_MODEL

        if self.provider == "xai":
            self.api_url = settings.LLM_API_URL or "https://api.x.ai/v1/chat/completions"
            self.model = settings.LLM_MODEL or "grok-4-fast-non-reasoning"
        elif self.provider == "anthropic":
            self.api_url = settings.LLM_API_URL or "https://api.anthropic.com/v1/messages"
            self.model = settings.LLM_MODEL or "claude-sonnet-4-20250514"
        elif self.provider == "openrouter":
            self.api_url = settings.LLM_API_URL or "https://openrouter.ai/api/v1/chat/completions"
            self.model = settings.LLM_MODEL or "openai/gpt-4o-mini"
        elif self.provider == "groq":
            self.api_url = settings.LLM_API_URL or "https://api.groq.com/openai/v1/chat/completions"
            self.model = settings.LLM_MODEL or "llama-3.3-70b-versatile"
        else:
            self.api_url = settings.LLM_API_URL or "https://api.openai.com/v1/chat/completions"
            self.model = settings.LLM_MODEL or "gpt-4o-mini"

        # Shared async client — reused across requests to benefit from connection pooling.
        # Limits: max 10 connections per host, keepalive 30 s, connect timeout 10 s.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=30),
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()

    @property
    def is_configured(self) -> bool:
        return bool(self.provider and self.api_key and self.model)

    @property
    def provider_name(self) -> str:
        mapping = {
            "anthropic": "Anthropic",
            "openai": "OpenAI",
            "openrouter": "OpenRouter",
            "xai": "xAI Grok",
            "groq": "Groq",
        }
        return mapping.get(self.provider, self.provider.title() if self.provider else "Unconfigured")

    @property
    def mode_name(self) -> str:
        if self.provider == "anthropic":
            return "live-multimodal"
        if self.provider in {"openai", "openrouter", "xai", "groq"}:
            return "live-chat-completions"
        return "live"

    def diagnostic_snapshot(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "provider_id": self.provider,
            "configured": self.is_configured,
            "model": self.model,
            "vision_model": self._vision_model_name(),
            "api_url": self.api_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "supports_image_inputs": self.provider in {"anthropic", "openai", "openrouter", "xai", "groq"},
        }

    async def generate_structured_analysis(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachment_bytes: Optional[bytes] = None,
        attachment_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_configured:
            return None

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "anthropic":
                    return await self._generate_with_anthropic(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        attachment_bytes=attachment_bytes,
                        attachment_type=attachment_type,
                    )
                if self.provider in {"openai", "openrouter", "xai", "groq"}:
                    return await self._generate_with_openai_compatible(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        attachment_bytes=attachment_bytes,
                        attachment_type=attachment_type,
                    )
                raise LLMServiceError(f"Unsupported LLM_PROVIDER '{self.provider}'.")
            except (httpx.HTTPError, ValueError, LLMServiceError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    last_error = LLMServiceError(_format_http_status_error(exc, self.provider))
                    if not _should_retry_http_status(exc):
                        break
                elif isinstance(exc, httpx.HTTPError):
                    last_error = LLMServiceError(f"Network error while calling {self.provider_name}: {exc}")
                else:
                    last_error = exc

                # Exponential backoff: 1 s, 2 s, 4 s … capped at 16 s
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2**attempt, 16))

        raise LLMServiceError(str(last_error) if last_error else "The LLM request failed.")

    async def _generate_with_anthropic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachment_bytes: Optional[bytes],
        attachment_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        content = [{"type": "text", "text": user_prompt}]
        if attachment_bytes and attachment_type:
            encoded = base64.b64encode(attachment_bytes).decode("utf-8")
            if attachment_type.startswith("image/"):
                content.insert(
                    0,
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": attachment_type,
                            "data": encoded,
                        },
                    },
                )
            elif attachment_type == "application/pdf":
                content.insert(
                    0,
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": encoded,
                        },
                    },
                )

        payload = {
            "model": self._model_for_request(attachment_type),
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = await self._http_client.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

        text_chunks = [
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        ]
        parsed = _extract_json_payload("\n".join(text_chunks))
        return parsed if isinstance(parsed, dict) else None

    async def _generate_with_openai_compatible(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachment_bytes: Optional[bytes],
        attachment_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        user_content: list[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if attachment_bytes and attachment_type and attachment_type.startswith("image/"):
            encoded = base64.b64encode(attachment_bytes).decode("utf-8")
            user_content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{attachment_type};base64,{encoded}"},
                },
            )

        payload = {
            "model": self._model_for_request(attachment_type),
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = settings.PROJECT_NAME

        response = await self._http_client.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

        choices = body.get("choices", [])
        if not choices:
            raise LLMServiceError("No completion choices were returned by the model.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
            raw_text = "\n".join(text_chunks)
        else:
            raw_text = str(content)

        parsed = _extract_json_payload(raw_text)
        return parsed if isinstance(parsed, dict) else None

    def _model_for_request(self, attachment_type: Optional[str]) -> str:
        if attachment_type and attachment_type.startswith("image/"):
            return self._vision_model_name()
        return self.model

    def _vision_model_name(self) -> str:
        if self.vision_model:
            return self.vision_model
        if self.provider == "openai":
            return "gpt-4o-mini"
        if self.provider == "openrouter":
            return "openai/gpt-4o-mini"
        if self.provider == "groq":
            return "meta-llama/llama-4-scout-17b-16e-instruct"
        return self.model


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from a model response.

    Handles three common shapes:
      1. Bare JSON object  { ... }
      2. Fenced block  ```json\\n{ ... }\\n```
      3. JSON embedded in prose  … { ... } …
    """
    cleaned = raw_text.strip()

    # Strip code fences (``` or ```json ... ```)
    if cleaned.startswith("```"):
        # Remove opening fence line
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[start : end + 1])


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    aliases = {
        "grok": "xai",
        "xai": "xai",
    }
    return aliases.get(normalized, normalized)


def _format_http_status_error(exc: httpx.HTTPStatusError, provider: str) -> str:
    response = exc.response
    details = _extract_error_message(response)
    status = response.status_code

    if provider == "openai" and status == 401:
        hint = (
            "OpenAI rejected the request with 401 Unauthorized. Check that LLM_API_KEY is set to a valid "
            "OpenAI API key and that the backend was restarted after updating backend/.env."
        )
        if details:
            return f"{hint} Provider response: {details}"
        return hint

    if provider == "openai" and status == 429:
        if _looks_like_quota_issue(details):
            hint = (
                "OpenAI rejected the request because this API project has no available quota. "
                "Add billing or prepaid credits in the OpenAI API project, then retry."
            )
        else:
            hint = (
                "OpenAI rejected the request with 429 Too Many Requests. This usually means the API project has "
                "hit a rate limit. Wait briefly or reduce request volume, then retry."
            )
        if details:
            return f"{hint} Provider response: {details}"
        return hint

    if provider == "xai" and status == 403:
        hint = (
            "xAI rejected the request with 403 Forbidden. This usually means the API key or team "
            "does not have permission, the key was revoked, or the xAI account needs billing/credits enabled."
        )
        if details:
            return f"{hint} Provider response: {details}"
        return hint

    if details:
        return f"HTTP {status} from {response.request.url}: {details}"
    return f"HTTP {status} from {response.request.url}"


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("error")
            if message:
                return str(message)
        message = payload.get("message")
        if message:
            return str(message)

    return json.dumps(payload)


def _looks_like_quota_issue(details: str) -> bool:
    lowered = details.lower()
    return "quota" in lowered or "billing" in lowered or "insufficient_quota" in lowered


def _should_retry_http_status(exc: httpx.HTTPStatusError) -> bool:
    status = exc.response.status_code
    details = _extract_error_message(exc.response)

    if status in {400, 401, 403, 404, 422}:
        return False
    if status == 429 and _looks_like_quota_issue(details):
        return False
    if 500 <= status < 600:
        return True
    return status == 429
