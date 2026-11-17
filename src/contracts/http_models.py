"""HTTP envelope extensions kept outside the calibration-bound local contract."""
from pydantic import BaseModel, ConfigDict, model_validator

from src.contracts.models import Category, ParseResponse


class SearchResponse(ParseResponse):
    category: Category | None

    @model_validator(mode="after")
    def non_search(self):
        if self.category is None and self.params:
            raise ValueError("non-search results require empty params")
        return self


class ProviderErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str


class ProviderErrorResponse(BaseModel):
    detail: ProviderErrorDetail
