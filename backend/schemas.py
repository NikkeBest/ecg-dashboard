from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserRegister(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(min_length=8)
    role: Literal["patient", "doctor"]


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    role: str
    approval_status: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ApprovalDecision(BaseModel):
    rejection_reason: Optional[str] = Field(default=None, max_length=1000)


class PredictionCreate(BaseModel):
    heartbeat_class: Literal["N", "S", "V", "F", "Q"]
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    heartbeat_time: Optional[float] = Field(default=None, ge=0)


class RecordingCreate(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    sampling_rate: int = Field(default=360, gt=0)
    started_at: Optional[datetime] = None
    predictions: List[PredictionCreate] = Field(default_factory=list)


class PredictionResponse(PredictionCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    recording_id: int


class RecordingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    patient_id: int
    device_id: Optional[int] = None
    file_name: Optional[str] = None
    sampling_rate: int
    started_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    has_waveform: bool = False
    lead_name: Optional[str] = None
    sample_count: Optional[int] = None


class RecordingDetailResponse(RecordingResponse):
    predictions: List[PredictionResponse] = Field(default_factory=list)


class AssignmentCreate(BaseModel):
    doctor_id: int = Field(gt=0)


class RecordingShareResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    recording_id: int
    patient_id: int
    doctor_id: int
    is_active: bool
    shared_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class ClinicalNoteCreate(BaseModel):
    note: str = Field(min_length=1, max_length=10000)
    patient_visible: bool = True


class ClinicalNoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    doctor_id: int
    patient_id: int
    recording_id: Optional[int] = None
    note: str
    patient_visible: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WaveformMetadataResponse(BaseModel):
    recording_id: int
    lead_name: Optional[str] = None
    sampling_rate: int
    sample_count: int
    has_waveform: bool


class AIAnalysisCreate(BaseModel):
    report_text: str = Field(min_length=1, max_length=30000)


class AIAnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    recording_id: int
    patient_id: int
    requested_by_id: int
    audience_role: str
    report_text: str
    created_at: Optional[datetime] = None


class ChatMessageCreate(BaseModel):
    patient_id: int = Field(gt=0)
    message_text: str = Field(min_length=1, max_length=5000)
    recording_id: Optional[int] = Field(default=None, gt=0)


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    patient_id: int
    doctor_id: int
    sender_id: int
    recording_id: Optional[int] = None
    message_text: str
    created_at: Optional[datetime] = None
