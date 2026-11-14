"""Run from the backend directory: python create_admin.py"""
from getpass import getpass
from database import SessionLocal
from models import User
from auth import hash_password

db = SessionLocal()
try:
    name = input("Admin full name: ").strip()
    email = input("Admin email: ").strip().lower()
    password = getpass("Admin password (minimum 12 characters): ")
    confirm = getpass("Confirm password: ")

    if not name or not email:
        raise SystemExit("Name and email are required.")
    if len(password) < 12:
        raise SystemExit("Use an admin password of at least 12 characters.")
    if password != confirm:
        raise SystemExit("Passwords do not match.")
    if db.query(User).filter(User.email == email).first():
        raise SystemExit("That email is already registered. No account was changed.")

    admin = User(
        full_name=name,
        email=email,
        password_hash=hash_password(password),
        role="admin",
        is_active=True,
        approval_status="approved",
    )
    db.add(admin)
    db.commit()
    print("Admin account created and approved.")
finally:
    db.close()
