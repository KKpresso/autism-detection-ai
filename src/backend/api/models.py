from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ModalityType(str, Enum):
    FMRI = "fmri"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


class TextInput(BaseModel):
    content: str = Field(..., description="Text content for analysis")
    language: str = Field(default="en", description="Language of the text")


class AudioFeatures(BaseModel):
    prosody_score: float
    rhythm_score: float
    pitch_variation: float
    speech_rate: float


class VideoFeatures(BaseModel):
    eye_contact_score: float
    facial_expression_score: float
    emotion_variation: float
    gaze_patterns: List[float]


class AnalysisResult(BaseModel):
    modality: ModalityType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Low/Medium/High risk assessment")
    features: dict = Field(..., description="Modality-specific features")
    timestamp: str = Field(..., description="Analysis timestamp")


class MultimodalPrediction(BaseModel):
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    modality_results: List[AnalysisResult]
    recommendation: str
    timestamp: str
