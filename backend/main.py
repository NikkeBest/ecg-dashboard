from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import engine, Base, get_db
from models import User, Recording, Prediction, DoctorPatient, RecordingShare, ClinicalNote, AIAnalysis, ChatMessage
from schemas import (
    UserRegister, UserLogin, UserResponse, TokenResponse,
    ApprovalDecision, AssignmentCreate, RecordingCreate, RecordingResponse,
    RecordingDetailResponse, PredictionResponse, RecordingShareResponse,
    ClinicalNoteCreate, ClinicalNoteResponse, WaveformMetadataResponse, AIAnalysisCreate, AIAnalysisResponse,
    ChatMessageCreate, ChatMessageResponse,
)
from auth import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from pathlib import Path
from uuid import uuid4
from fastapi.responses import FileResponse

app = FastAPI(
    title="ECG Monitoring System",
    description="Admin, Patient and Doctor ECG Monitoring Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise credentials_error
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active or user.approval_status != "approved":
        raise credentials_error
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return current_user


def can_access_recording(user: User, recording: Recording, db: Session) -> bool:
    if user.role == "patient":
        return recording.patient_id == user.id
    if user.role == "doctor":
        assigned = db.query(DoctorPatient).filter(
            DoctorPatient.doctor_id == user.id,
            DoctorPatient.patient_id == recording.patient_id,
        ).first()
        shared = db.query(RecordingShare).filter(
            RecordingShare.recording_id == recording.id,
            RecordingShare.patient_id == recording.patient_id,
            RecordingShare.doctor_id == user.id,
            RecordingShare.is_active.is_(True),
        ).first()
        return assigned is not None and shared is not None
    if user.role == "admin":
        return True
    return False


@app.get("/")
def home():
    return {"message": "ECG Monitoring Backend is running!"}


@app.get("/database-status")
def database_status():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"database": "Connected", "status": "Working"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database connection failed.")


@app.get("/tables-status")
def tables_status():
    return {"tables": list(Base.metadata.tables.keys()), "status": "Created"}


@app.post("/auth/register", response_model=UserResponse, status_code=201)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email is already registered.")
    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        role=user_data.role,
        is_active=True,
        approval_status="pending",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/auth/login", response_model=TokenResponse)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive.")
    if user.approval_status == "pending":
        raise HTTPException(status_code=403, detail="Your account is waiting for administrator approval.")
    if user.approval_status == "rejected":
        raise HTTPException(status_code=403, detail="Your registration was rejected. Contact the administrator.")
    if user.approval_status != "approved":
        raise HTTPException(status_code=403, detail="Account is not approved.")
    return {
        "access_token": create_access_token(user.id, user.role),
        "token_type": "bearer",
        "user": user,
    }

@app.post("/auth/token")
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    if not user.is_active or user.approval_status != "approved":
        raise HTTPException(status_code=403, detail="Account is not approved or active.")

    return {
        "access_token": create_access_token(user.id, user.role),
        "token_type": "bearer"
    }


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# Admin-only registration approval endpoints
@app.get("/admin/pending-users", response_model=List[UserResponse])
def pending_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    return db.query(User).filter(
        User.role.in_(["patient", "doctor"]),
        User.approval_status == "pending",
    ).order_by(User.created_at.asc()).all()


@app.get("/admin/users", response_model=List[UserResponse])
def list_users(
    role: str = "patient",
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if role not in {"patient", "doctor", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role filter.")
    return db.query(User).filter(User.role == role).order_by(User.created_at.desc()).all()


@app.post("/admin/users/{user_id}/approve", response_model=UserResponse)
def approve_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.role not in {"patient", "doctor"}:
        raise HTTPException(status_code=404, detail="Patient or doctor account not found.")
    if user.approval_status == "approved":
        return user
    user.approval_status = "approved"
    user.is_active = True
    user.approved_by = admin.id
    user.approved_at = datetime.now(timezone.utc)
    user.rejection_reason = None
    db.commit()
    db.refresh(user)
    return user


@app.post("/admin/users/{user_id}/reject", response_model=UserResponse)
def reject_user(
    user_id: int,
    decision: ApprovalDecision,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.role not in {"patient", "doctor"}:
        raise HTTPException(status_code=404, detail="Patient or doctor account not found.")
    user.approval_status = "rejected"
    user.is_active = False
    user.approved_by = admin.id
    user.approved_at = datetime.now(timezone.utc)
    user.rejection_reason = decision.rejection_reason
    db.commit()
    db.refresh(user)
    return user



# Admin-managed one-doctor-per-patient assignments.
@app.get("/admin/assignments")
def list_assignments(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    from sqlalchemy.orm import aliased
    Patient = aliased(User)
    Doctor = aliased(User)
    rows = (
        db.query(DoctorPatient, Patient, Doctor)
        .join(Patient, DoctorPatient.patient_id == Patient.id)
        .join(Doctor, DoctorPatient.doctor_id == Doctor.id)
        .order_by(Patient.full_name.asc())
        .all()
    )
    return [
        {"patient_id": a.patient_id, "patient_name": p.full_name,
         "patient_email": p.email, "doctor_id": a.doctor_id,
         "doctor_name": d.full_name, "doctor_email": d.email}
        for a, p, d in rows
    ]


@app.put("/admin/patients/{patient_id}/assignment")
def assign_doctor_to_patient(
    patient_id: int,
    assignment_data: AssignmentCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    patient = db.query(User).filter(
        User.id == patient_id, User.role == "patient", User.is_active.is_(True),
        User.approval_status == "approved",
    ).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Approved active patient not found.")
    doctor = db.query(User).filter(
        User.id == assignment_data.doctor_id, User.role == "doctor", User.is_active.is_(True),
        User.approval_status == "approved",
    ).first()
    if doctor is None:
        raise HTTPException(status_code=404, detail="Approved active doctor not found.")
    assignment = db.query(DoctorPatient).filter(DoctorPatient.patient_id == patient_id).first()
    try:
        if assignment is None:
            assignment = DoctorPatient(patient_id=patient_id, doctor_id=doctor.id)
            db.add(assignment)
        else:
            assignment.doctor_id = doctor.id
        db.commit()
        db.refresh(assignment)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="Assignment could not be saved. Refresh and try again.")
    return {"message": "Doctor assigned successfully.", "patient_id": patient.id,
            "patient_name": patient.full_name, "doctor_id": doctor.id, "doctor_name": doctor.full_name}


@app.delete("/admin/patients/{patient_id}/assignment")
def remove_patient_assignment(
    patient_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    assignment = db.query(DoctorPatient).filter(DoctorPatient.patient_id == patient_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Patient has no assigned doctor.")
    db.delete(assignment)
    db.commit()
    return {"message": "Patient assignment removed.", "patient_id": patient_id}



@app.get("/doctor/patients", response_model=List[UserResponse])
def get_doctor_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor account required.")
    return (db.query(User)
        .join(DoctorPatient, DoctorPatient.patient_id == User.id)
        .filter(DoctorPatient.doctor_id == current_user.id,
                User.role == "patient", User.is_active.is_(True),
                User.approval_status == "approved")
        .order_by(User.full_name.asc()).all())


@app.get("/patients/my-doctor")
def get_my_doctor(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient account required.")
    from sqlalchemy.orm import aliased
    Doctor = aliased(User)
    row = (db.query(DoctorPatient, Doctor)
        .join(Doctor, DoctorPatient.doctor_id == Doctor.id)
        .filter(DoctorPatient.patient_id == current_user.id).first())
    if row is None:
        return {"assigned": False, "doctor": None}
    _, doctor = row
    return {"assigned": True, "doctor": {"id": doctor.id, "full_name": doctor.full_name, "email": doctor.email}}


# ECG metadata and prediction endpoints retained for the existing patient workflow.
@app.post("/recordings/{recording_id}/share", response_model=RecordingShareResponse)
def share_recording_with_assigned_doctor(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only the patient who owns a recording can share it.")
    recording = db.query(Recording).filter(
        Recording.id == recording_id, Recording.patient_id == current_user.id
    ).first()
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found.")
    assignment = db.query(DoctorPatient).filter(
        DoctorPatient.patient_id == current_user.id
    ).first()
    if assignment is None:
        raise HTTPException(status_code=409, detail="You do not currently have an assigned doctor.")
    doctor = db.query(User).filter(
        User.id == assignment.doctor_id, User.role == "doctor",
        User.is_active.is_(True), User.approval_status == "approved",
    ).first()
    if doctor is None:
        raise HTTPException(status_code=409, detail="Your assigned doctor is not currently available.")
    share = db.query(RecordingShare).filter(RecordingShare.recording_id == recording.id).first()
    if share is None:
        share = RecordingShare(recording_id=recording.id, patient_id=current_user.id,
                               doctor_id=doctor.id, is_active=True, revoked_at=None)
        db.add(share)
    else:
        share.patient_id = current_user.id
        share.doctor_id = doctor.id
        share.is_active = True
        share.shared_at = datetime.now(timezone.utc)
        share.revoked_at = None
    db.commit(); db.refresh(share)
    return share


@app.delete("/recordings/{recording_id}/share", response_model=RecordingShareResponse)
def revoke_recording_share(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only the patient who owns a recording can revoke sharing.")
    share = db.query(RecordingShare).filter(
        RecordingShare.recording_id == recording_id,
        RecordingShare.patient_id == current_user.id,
    ).first()
    if share is None or not share.is_active:
        raise HTTPException(status_code=404, detail="No active share exists for this recording.")
    share.is_active = False
    share.revoked_at = datetime.now(timezone.utc)
    db.commit(); db.refresh(share)
    return share


@app.get("/recordings/{recording_id}/share", response_model=RecordingShareResponse)
def get_recording_share_status(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient account required.")
    recording = db.query(Recording).filter(
        Recording.id == recording_id, Recording.patient_id == current_user.id
    ).first()
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found.")
    share = db.query(RecordingShare).filter(RecordingShare.recording_id == recording_id).first()
    if share is None:
        raise HTTPException(status_code=404, detail="This recording has not been shared.")
    return share



# Server-side storage for normalized ECG waveform arrays.
WAVEFORM_DIR = Path(__file__).resolve().parent / "ecg_waveforms"
WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)
MAX_WAVEFORM_BYTES = 50 * 1024 * 1024


@app.put("/recordings/{recording_id}/waveform", response_model=WaveformMetadataResponse)
async def upload_recording_waveform(
    recording_id: int,
    waveform_file: UploadFile = File(...),
    lead_name: str = Form("Lead 0"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only the patient who owns a recording can upload its waveform.")
    recording = db.query(Recording).filter(
        Recording.id == recording_id, Recording.patient_id == current_user.id
    ).first()
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found.")
    if waveform_file.content_type not in {"application/octet-stream", "application/x-npz", "application/zip"}:
        # Browser/client MIME values vary; the file is still validated as a NumPy archive below.
        pass
    raw = await waveform_file.read(MAX_WAVEFORM_BYTES + 1)
    if not raw or len(raw) > MAX_WAVEFORM_BYTES:
        raise HTTPException(status_code=413, detail="Waveform file is empty or exceeds the 50 MB limit.")
    target = WAVEFORM_DIR / f"recording_{recording_id}_{uuid4().hex}.npz"
    temp = target.with_suffix(".tmp")
    try:
        import io
        import numpy as np
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if "signal" not in archive or "sampling_rate" not in archive:
                raise ValueError("NPZ must contain signal and sampling_rate arrays.")
            signal = np.asarray(archive["signal"], dtype=np.float32).reshape(-1)
            fs = int(np.asarray(archive["sampling_rate"]).reshape(-1)[0])
        if signal.size < 2 or signal.size > 20_000_000 or fs <= 0:
            raise ValueError("Waveform dimensions or sampling rate are invalid.")
        if not np.isfinite(signal).all():
            raise ValueError("Waveform contains non-finite samples.")
        # Re-serialize a validated archive to avoid persisting arbitrary uploaded ZIP content.
        with open(temp, "wb") as handle:
            np.savez_compressed(handle, signal=signal, sampling_rate=np.array([fs], dtype=np.int32))
        temp.replace(target)
    except Exception as exc:
        if temp.exists():
            temp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid ECG waveform upload: {exc}")
    old_path = Path(recording.waveform_path) if recording.waveform_path else None
    recording.waveform_path = str(target)
    recording.lead_name = (lead_name or "Lead 0")[:100]
    recording.sample_count = int(signal.size)
    recording.sampling_rate = fs
    db.commit()
    if old_path and old_path.exists() and WAVEFORM_DIR in old_path.parents:
        old_path.unlink(missing_ok=True)
    return {
        "recording_id": recording.id,
        "lead_name": recording.lead_name,
        "sampling_rate": recording.sampling_rate,
        "sample_count": recording.sample_count,
        "has_waveform": True,
    }


@app.get("/recordings/{recording_id}/waveform")
def get_recording_waveform(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found.")
    if not recording.waveform_path:
        raise HTTPException(status_code=404, detail="No waveform has been stored for this recording.")
    path = Path(recording.waveform_path).resolve()
    if WAVEFORM_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Stored waveform file is unavailable.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"recording_{recording_id}.npz")


@app.get("/recordings/{recording_id}/waveform/metadata", response_model=WaveformMetadataResponse)
def get_recording_waveform_metadata(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found.")
    return {
        "recording_id": recording.id,
        "lead_name": recording.lead_name,
        "sampling_rate": recording.sampling_rate,
        "sample_count": recording.sample_count or 0,
        "has_waveform": bool(recording.waveform_path and Path(recording.waveform_path).is_file()),
    }


@app.get("/recordings/{recording_id}/notes", response_model=List[ClinicalNoteResponse])
def list_recording_notes(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found.")
    query = db.query(ClinicalNote).filter(ClinicalNote.recording_id == recording_id)
    if current_user.role == "patient":
        query = query.filter(
            ClinicalNote.patient_id == current_user.id,
            ClinicalNote.patient_visible.is_(True),
        )
    elif current_user.role == "doctor":
        query = query.filter(ClinicalNote.doctor_id == current_user.id)
    return query.order_by(ClinicalNote.created_at.desc()).all()


@app.post("/recordings/{recording_id}/notes", response_model=ClinicalNoteResponse, status_code=201)
def create_recording_note(
    recording_id: int,
    note_data: ClinicalNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can create clinical notes.")
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Shared recording not found.")
    note_text = note_data.note.strip()
    if not note_text:
        raise HTTPException(status_code=422, detail="Clinical note cannot be empty.")
    note = ClinicalNote(
        doctor_id=current_user.id,
        patient_id=recording.patient_id,
        recording_id=recording.id,
        note=note_text,
        patient_visible=note_data.patient_visible,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@app.post("/recordings", response_model=RecordingDetailResponse, status_code=201)
def create_recording(
    recording_data: RecordingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patient accounts can submit ECG recordings.")
    recording = Recording(
        patient_id=current_user.id,
        file_path=recording_data.file_name,
        sampling_rate=recording_data.sampling_rate,
        started_at=recording_data.started_at,
    )
    try:
        db.add(recording)
        db.flush()
        rows = [
            Prediction(
                recording_id=recording.id,
                heartbeat_class=item.heartbeat_class,
                confidence=item.confidence,
                heartbeat_time=item.heartbeat_time,
            )
            for item in recording_data.predictions
        ]
        db.add_all(rows)
        db.commit()
        db.refresh(recording)
        for row in rows:
            db.refresh(row)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not save ECG recording and predictions.")
    return {
        "id": recording.id,
        "patient_id": recording.patient_id,
        "device_id": recording.device_id,
        "file_name": recording.file_path,
        "sampling_rate": recording.sampling_rate,
        "started_at": recording.started_at,
        "created_at": recording.created_at,
        "predictions": rows,
    }


@app.get("/recordings", response_model=List[RecordingResponse])
def list_recordings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Recording)
    if current_user.role == "patient":
        query = query.filter(Recording.patient_id == current_user.id)
    elif current_user.role == "doctor":
        shared_ids = db.query(RecordingShare.recording_id).join(
            DoctorPatient,
            (DoctorPatient.patient_id == RecordingShare.patient_id)
            & (DoctorPatient.doctor_id == RecordingShare.doctor_id),
        ).filter(
            RecordingShare.doctor_id == current_user.id,
            RecordingShare.is_active.is_(True),
        ).subquery()
        query = query.filter(Recording.id.in_(shared_ids))
    elif current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Role is not permitted.")
    rows = query.order_by(Recording.created_at.desc()).all()
    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "device_id": row.device_id,
            "file_name": row.file_path,
            "sampling_rate": row.sampling_rate,
            "started_at": row.started_at,
            "created_at": row.created_at,
            "has_waveform": bool(row.waveform_path and Path(row.waveform_path).is_file()),
            "lead_name": row.lead_name,
            "sample_count": row.sample_count,
        }
        for row in rows
    ]


@app.get("/recordings/{recording_id}", response_model=RecordingDetailResponse)
def get_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found.")
    predictions = db.query(Prediction).filter(
        Prediction.recording_id == recording.id
    ).order_by(Prediction.heartbeat_time.asc(), Prediction.id.asc()).all()
    return {
        "id": recording.id,
        "patient_id": recording.patient_id,
        "device_id": recording.device_id,
        "file_name": recording.file_path,
        "sampling_rate": recording.sampling_rate,
        "started_at": recording.started_at,
        "created_at": recording.created_at,
        "predictions": predictions,
    }


@app.get("/recordings/{recording_id}/predictions", response_model=List[PredictionResponse])
def list_predictions(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found.")
    return db.query(Prediction).filter(
        Prediction.recording_id == recording_id
    ).order_by(Prediction.heartbeat_time.asc(), Prediction.id.asc()).all()


# Role-scoped AI report history. Reports are decision-support text, not diagnoses.
@app.post("/recordings/{recording_id}/ai-analyses", response_model=AIAnalysisResponse, status_code=201)
def save_ai_analysis(
    recording_id: int,
    analysis_data: AIAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"patient", "doctor"}:
        raise HTTPException(status_code=403, detail="Patient or doctor account required.")
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found or access not shared.")
    report = analysis_data.report_text.strip()
    if not report:
        raise HTTPException(status_code=422, detail="AI report cannot be empty.")
    row = AIAnalysis(
        recording_id=recording.id,
        patient_id=recording.patient_id,
        requested_by_id=current_user.id,
        audience_role=current_user.role,
        report_text=report,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@app.get("/recordings/{recording_id}/ai-analyses", response_model=List[AIAnalysisResponse])
def list_ai_analyses(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_recording(current_user, recording, db):
        raise HTTPException(status_code=404, detail="Recording not found or access not shared.")
    return (
        db.query(AIAnalysis)
        .filter(
            AIAnalysis.recording_id == recording_id,
            AIAnalysis.requested_by_id == current_user.id,
            AIAnalysis.audience_role == current_user.role,
        )
        .order_by(AIAnalysis.created_at.desc())
        .all()
    )


# Persistent patient-doctor chat. Only the currently assigned pair can participate.
def _chat_pair(patient_id: int, current_user: User, db: Session):
    assignment = db.query(DoctorPatient).filter(DoctorPatient.patient_id == patient_id).first()
    if assignment is None:
        raise HTTPException(status_code=403, detail="This patient does not currently have an assigned doctor.")
    if current_user.role == "patient":
        if current_user.id != patient_id:
            raise HTTPException(status_code=403, detail="You can only access your own chat.")
        doctor_id = assignment.doctor_id
    elif current_user.role == "doctor":
        if assignment.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="You are not the patient's currently assigned doctor.")
        doctor_id = current_user.id
    else:
        raise HTTPException(status_code=403, detail="Only patients and doctors can use chat.")
    doctor = db.query(User).filter(User.id == doctor_id, User.role == "doctor", User.is_active.is_(True), User.approval_status == "approved").first()
    patient = db.query(User).filter(User.id == patient_id, User.role == "patient", User.is_active.is_(True), User.approval_status == "approved").first()
    if not doctor or not patient:
        raise HTTPException(status_code=403, detail="The patient or assigned doctor account is unavailable.")
    return doctor_id


def _validate_chat_recording(patient_id: int, doctor_id: int, recording_id: int, db: Session):
    recording = db.query(Recording).filter(Recording.id == recording_id, Recording.patient_id == patient_id).first()
    share = db.query(RecordingShare).filter(
        RecordingShare.recording_id == recording_id, RecordingShare.patient_id == patient_id,
        RecordingShare.doctor_id == doctor_id, RecordingShare.is_active.is_(True),
    ).first()
    if recording is None or share is None:
        raise HTTPException(status_code=403, detail="That ECG record is not currently shared with the assigned doctor.")


@app.get("/chat/{patient_id}/messages", response_model=List[ChatMessageResponse])
def get_chat_messages(
    patient_id: int, recording_id: int | None = None,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doctor_id = _chat_pair(patient_id, current_user, db)
    query = db.query(ChatMessage).filter(ChatMessage.patient_id == patient_id, ChatMessage.doctor_id == doctor_id)
    if recording_id is not None:
        _validate_chat_recording(patient_id, doctor_id, recording_id, db)
        query = query.filter(ChatMessage.recording_id == recording_id)
    else:
        query = query.filter(ChatMessage.recording_id.is_(None))
    return query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).limit(500).all()


@app.post("/chat/messages", response_model=ChatMessageResponse, status_code=201)
def send_chat_message(
    message_data: ChatMessageCreate, db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doctor_id = _chat_pair(message_data.patient_id, current_user, db)
    message = message_data.message_text.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")
    if message_data.recording_id is not None:
        _validate_chat_recording(message_data.patient_id, doctor_id, message_data.recording_id, db)
    row = ChatMessage(
        patient_id=message_data.patient_id, doctor_id=doctor_id, sender_id=current_user.id,
        recording_id=message_data.recording_id, message_text=message,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
