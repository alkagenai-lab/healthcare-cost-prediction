from enum import Enum

from pydantic import BaseModel, Field


class SexEnum(str, Enum):
    """Allowed patient sex values."""

    MALE = "male"
    FEMALE = "female"


class SmokerEnum(str, Enum):
    """Allowed smoking status values."""

    YES = "yes"
    NO = "no"


class RegionEnum(str, Enum):
    """Allowed residential regions."""

    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


class PredictionRequest(BaseModel):
    """Request schema for healthcare cost prediction."""

    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age in years.",
    )

    sex: SexEnum = Field(
        ...,
        description="Patient sex.",
    )

    bmi: float = Field(
        ...,
        gt=0,
        le=100,
        description="Body Mass Index.",
    )

    children: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of children/dependents.",
    )

    smoker: SmokerEnum = Field(
        ...,
        description="Smoking status.",
    )

    region: RegionEnum = Field(
        ...,
        description="Residential region.",
    )