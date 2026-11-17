"""Process configuration; environment variables take precedence over .env."""
from pathlib import Path
from typing import Literal
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extractor_enabled: bool = False
    parser_mode: Literal["stub", "local", "openai", "combined"] | None = None
    extractor_use_tagger: bool = True
    classifier_artifact_path: Path = ROOT / "src/artifacts/classifier.joblib"
    e5_model_path: Path = ROOT / "models/multilingual-e5-small"
    confidence_artifact_path: Path = ROOT / "src/artifacts/confidence.joblib"
    parse_confidence_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    # which local parser serves when extractor_enabled: "frozen_fusion" (default) or "tagger" (src/tagger)
    local_parser: Literal["frozen_fusion", "tagger"] = "frozen_fusion"
    tagger_model_dir: Path = ROOT / "models/fine-tuned-tagger"
    tagger_artifact_dir: Path = ROOT / "src/artifacts/tagger"
    tagger_gate_path: Path = ROOT / "src/artifacts/tagger/gate.joblib"
    # product rule: a hybrid powertrain also sets סוגי_רכב; off by default (the dataset labels do not follow it)
    tagger_hybrid_vehicle_type: bool = False
    openai_fallback_enabled: bool = False
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_reasoning_effort: str | None = None
    openai_few_shot_count: Literal[0, 6, 12] = 0
    openai_guidelines_enabled: bool = True
    expose_token_usage: bool = False
    openai_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    openai_max_output_tokens: int = Field(default=4096, gt=0)

    @field_validator("extractor_enabled", "extractor_use_tagger", "tagger_hybrid_vehicle_type",
                     "openai_fallback_enabled", "openai_guidelines_enabled", "expose_token_usage", mode="before")
    @classmethod
    def boolean(cls, value):
        if isinstance(value, str):
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("expected true, false, 1, or 0")
            return value.lower() in {"true", "1"}
        return value

    @field_validator("openai_model", "openai_reasoning_effort")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("must not be blank; omit optional settings instead")
        return value

    @field_validator("openai_few_shot_count", mode="before")
    @classmethod
    def example_count(cls, value):
        if isinstance(value, bool):
            raise ValueError("example count must be 0, 6, or 12")
        return int(value) if isinstance(value, str) and value.isdigit() else value

    @model_validator(mode="after")
    def openai_configuration(self):
        if self.openai_fallback_enabled or self.parser_mode in {"openai", "combined"}:
            if not self.openai_model or not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
                raise ValueError("OpenAI fallback requires OPENAI_MODEL and OPENAI_API_KEY")
        return self

    @property
    def active_parser_mode(self):
        if self.parser_mode is not None:
            return self.parser_mode
        if not self.extractor_enabled:
            return "stub"
        return "combined" if self.openai_fallback_enabled else "local"

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls.model_validate({k: os.environ[k.upper()] for k in cls.model_fields if k.upper() in os.environ})
