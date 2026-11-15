"""Public HTTP contracts and the typed stub fixture."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal["נדל״ן", "רכב", "יד_שנייה"]


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    q: str = Field(min_length=1, max_length=2000)

    @field_validator("q")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("q must not be blank")
        return value


class ParseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    category: Category
    params: dict[str, Any]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    notes: list[str] = Field(default_factory=list)


class StubPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    min: Literal[0] = 0
    max: Literal[1000000] = 1000000


class StubParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    property_types: list[Literal["דירה"]] = Field(alias="סוגי_נכס", min_length=1, max_length=1)
    rooms: Literal[3] = Field(alias="מס׳_חדרים")
    city: Literal["ירושלים"] = Field(alias="עיר")
    price: StubPrice = Field(alias="מחיר")
