"""Process configuration; environment variables take precedence over .env."""
from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extractor_enabled: bool = False
    extractor_use_tagger: bool = True
    classifier_artifact_path: Path = ROOT / "src/artifacts/classifier.joblib"
    e5_model_path: Path = ROOT / "models/multilingual-e5-small"
    confidence_artifact_path: Path = ROOT / "src/artifacts/confidence.joblib"
    parse_confidence_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("extractor_enabled", "extractor_use_tagger", mode="before")
    @classmethod
    def boolean(cls, value):
        if isinstance(value, str):
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("expected true, false, 1, or 0")
            return value.lower() in {"true", "1"}
        return value

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls.model_validate({k: os.environ[k.upper()] for k in cls.model_fields if k.upper() in os.environ})
