"""
Xac thuc Firebase + he thong credit (luu tren Firestore).

Vi sao Firestore: Render free tier co RAM/disk ephemeral (mat khi ngu day / restart),
KHONG the dung de luu so du credit -> mat tien cua user. Firestore ben vung, va du an
da dung Firebase Auth san o frontend nen dung luon cung project.

Cau hinh tren Render: set 1 trong 2 env:
  - FIREBASE_SERVICE_ACCOUNT_JSON = toan bo noi dung file service account JSON (1 dong)
  - hoac GOOGLE_APPLICATION_CREDENTIALS = duong dan toi file JSON (dung Application Default)

Lay service account: Firebase Console > Project Settings > Service accounts >
Generate new private key. (Khac voi firebaseConfig o frontend — cai do la public key.)
"""
import os
import json
import threading
from typing import Optional

from fastapi import Header, HTTPException

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore

# So credit tang free cho user moi (= so video tao thu mien phi). Doi qua env neu can.
FREE_CREDITS = int(os.getenv("FREE_CREDITS", "3"))

_db = None
_init_lock = threading.Lock()


def _get_db():
    """Khoi tao firebase-admin + Firestore client 1 lan (lazy, thread-safe).
    Neu chua cau hinh credentials -> nem loi ro rang (fail-safe: KHONG cho dung chua tra phi)."""
    global _db
    if _db is not None:
        return _db
    with _init_lock:
        if _db is not None:
            return _db
        if not firebase_admin._apps:
            sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            try:
                if sa_json:
                    cred = credentials.Certificate(json.loads(sa_json))
                else:
                    # Dua vao GOOGLE_APPLICATION_CREDENTIALS (Application Default Credentials)
                    cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Backend chua cau hinh Firebase (credit/auth tam ngung): {e}",
                )
        _db = firestore.client()
    return _db


def require_uid(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency: doc 'Authorization: Bearer <idToken>', verify -> tra uid.
    Dung lam cong chan: endpoint nao co dependency nay thi BAT BUOC dang nhap."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Cần đăng nhập để dùng tính năng này.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        # _get_db() de chac chan firebase_admin da initialize truoc khi verify token.
        _get_db()
        decoded = fb_auth.verify_id_token(token)
    except HTTPException:
        raise
    except Exception as e:
        # Kem ly do that (vd "incorrect audience" = service account khac project) de de chan doan.
        raise HTTPException(status_code=401, detail=f"Token không hợp lệ: {e}")
    return decoded["uid"]


def admin_project_id() -> str:
    """Tra ve project_id ma firebase-admin dang dung (de doi chieu voi project frontend).
    project_id KHONG phai bi mat -> an toan de lo ra cho chan doan."""
    try:
        _get_db()
        return firebase_admin.get_app().project_id or "(khong xac dinh)"
    except Exception as e:
        return f"(init that bai: {e})"


def get_credits(uid: str) -> int:
    """Doc so du credit. User moi (chua co doc) -> tao doc voi FREE_CREDITS."""
    db = _get_db()
    ref = db.collection("users").document(uid)
    snap = ref.get()
    if not snap.exists:
        ref.set({"credits": FREE_CREDITS})
        return FREE_CREDITS
    return int((snap.to_dict() or {}).get("credits", 0))


def reserve_credit(uid: str, amount: int = 1) -> bool:
    """Tru credit ATOMIC (transaction) truoc khi nhan job -> chong:
      - tru am / race condition khi bam nhanh nhieu lan,
      - 1 user xep hang loat job chi voi 1 credit.
    Tra True neu da tru thanh cong, False neu khong du credit."""
    db = _get_db()
    ref = db.collection("users").document(uid)

    @firestore.transactional
    def _txn(transaction):
        snap = ref.get(transaction=transaction)
        if snap.exists:
            bal = int((snap.to_dict() or {}).get("credits", 0))
        else:
            bal = FREE_CREDITS                     # user moi: tang san FREE_CREDITS
        if bal < amount:
            return False
        transaction.set(ref, {"credits": bal - amount}, merge=True)
        return True

    return _txn(db.transaction())


def refund_credit(uid: str, amount: int = 1) -> None:
    """Hoan credit (khi job render that bai). Tang nguyen tu bang Increment."""
    try:
        db = _get_db()
        db.collection("users").document(uid).set(
            {"credits": firestore.Increment(amount)}, merge=True
        )
    except Exception:
        # Hoan that bai khong duoc lam sap worker; chi log ngam.
        pass
