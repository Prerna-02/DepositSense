"""
Pydantic schemas for the Bank Marketing prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    """Input features for a single customer prediction."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type (e.g. admin., technician, management)")
    marital: str = Field(..., description="Marital status (married, single, divorced)")
    education: str = Field(..., description="Education level (primary, secondary, tertiary, unknown)")
    default: str = Field(..., description="Has credit in default? (yes/no)")
    balance: int = Field(..., description="Average yearly balance in euros")
    housing: str = Field(..., description="Has housing loan? (yes/no)")
    loan: str = Field(..., description="Has personal loan? (yes/no)")
    contact: str = Field(..., description="Contact communication type")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    month: str = Field(..., description="Last contact month of year (jan, feb, ...)")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., description="Days since last contact from previous campaign (-1 = not contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous campaign (success, failure, other, unknown)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "duration": 250,
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            }
        }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    probability: float = Field(..., description="Probability of subscribing (0-1)")
    prediction: str = Field(..., description="Predicted label: yes or no")
    model_version: str = Field(default="v1.0", description="Model version identifier")


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    model_loaded: bool
