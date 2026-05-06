import os
from pathlib import Path


def _load_dotenv() -> None:
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[2]
    candidate_files = [repo_root / ".env", repo_root / "backend" / ".env"]

    for env_file in candidate_files:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


class Settings:
    PROJECT_NAME: str = "MedTranslate AI"
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0").strip()
    API_PREFIX: str = ""
    # Comma-separated list of allowed CORS origins; defaults to localhost Angular dev server
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:4200,http://127.0.0.1:4200,https://transcendent-hamster-fb4b72.netlify.app",
        ).split(",")
        if o.strip()
    ]
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "").strip().lower()
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()
    LLM_VISION_MODEL: str = os.getenv("LLM_VISION_MODEL", "").strip()
    LLM_API_URL: str = os.getenv("LLM_API_URL", "").strip()
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1400"))
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_PROMPT_TEXT_LIMIT: int = int(os.getenv("LLM_PROMPT_TEXT_LIMIT", "12000"))
    LOCAL_XRAY_MODEL_PATH: str = os.getenv("LOCAL_XRAY_MODEL_PATH", "")
    LOCAL_XRAY_TOP_K: int = int(os.getenv("LOCAL_XRAY_TOP_K", "5"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))


settings = Settings()
