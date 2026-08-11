"""Local LLM backend — OpenAI, Gemini, or custom endpoint."""
from ..config import (
    CUSTOM_API_KEY,
    CUSTOM_BASE_URL,
    CUSTOM_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client_kwargs = {"api_key": require_openai_key()}
    if CUSTOM_BASE_URL:
        client_kwargs["base_url"] = CUSTOM_BASE_URL
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=CUSTOM_MODEL or OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_custom_llm(prompt: str) -> str:
    """Custom OpenAI-compatible endpoint (OpenRouter, Ollama, self-hosted, etc.)."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    if not CUSTOM_API_KEY:
        raise RuntimeError(
            "CUSTOM_API_KEY is not set. Set it in the environment or config to use a custom LLM endpoint."
        )
    if not CUSTOM_BASE_URL:
        raise RuntimeError(
            "CUSTOM_LLM_BASE_URL is not set. Provide your custom API base URL."
        )

    client = OpenAI(api_key=CUSTOM_API_KEY, base_url=CUSTOM_BASE_URL)
    model = CUSTOM_MODEL or "default"
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "custom":
        return call_custom_llm(prompt)
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'custom'."
    )
