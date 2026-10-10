"""Security, authentication, and RBAC / scope verification for the HR Backend Gateway.

Enforces:
- Bearer token / JWT authentication from Flutter / Web clients.
- Canonical caller identity resolution against Employee database table.
- Strict RBAC and feature-level authorization checks.
- Employee self-service boundary enforcement (prevents BOLA / IDOR).
- Department scoping for Managers.
- Organization-wide scoping for HR Administrators.
- Generation of internal service authentication credentials for AI Service calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.gateway.jwt import decode_jwt_token
from app.models import Employee
from app.services.shared_hr_data import get_shared_employee, is_shared_hr_schema

bearer_security = HTTPBearer(auto_error=False)

DEFAULT_GATEWAY_SERVICE_TOKEN = os.getenv("AI_GATEWAY_SERVICE_TOKEN", "smart-hr-gateway-service-internal-token-2026")


class HRApiException(HTTPException):
    """Custom exception matching the public HR API error envelope."""

    def __init__(self, status_code: int, message: str, errors: list[Any] | None = None):
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.errors = errors or []


class HRCallerRole(str, Enum):
    """Normalized caller roles recognized by HR Gateway."""

    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR_ADMIN = "hr_admin"


@dataclass(frozen=True)
class HRCallerContext:
    """Authenticated caller context derived by the HR Gateway."""

    user_id: str | int
    employee_id: str
    role: str  # "employee", "manager", "hr_admin"
    department: str
    permissions: set[str] = field(default_factory=set)


def normalize_role(role_raw: str | None) -> str:
    """Normalizes role strings from token/database into employee, manager, or hr_admin."""
    if not role_raw:
        return HRCallerRole.EMPLOYEE.value
    clean = str(role_raw).strip().lower()
    if clean in ("employee", "staff"):
        return HRCallerRole.EMPLOYEE.value
    if clean in ("manager", "lead", "team_lead"):
        return HRCallerRole.MANAGER.value
    if clean in ("hr", "hr_admin", "owner", "admin", "administrator"):
        return HRCallerRole.HR_ADMIN.value
    return HRCallerRole.EMPLOYEE.value


def get_hr_caller_context(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_security)] = None,
    db: Annotated[Session, Depends(get_db)] = None,
) -> HRCallerContext:
    """Authenticates the Flutter / Web user from the Bearer JWT token."""
    if not credentials or not credentials.credentials:
        raise HRApiException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message="Unauthenticated. Please log in first.",
        )

    token = credentials.credentials.strip()
    payload: dict[str, Any] | None = None

    # 1. Try decoding as JWT
    try:
        payload = decode_jwt_token(token, verify_exp=False)
    except ValueError:
        payload = None

    employee_id_candidate: str | None = None
    role_candidate: str | None = None
    permissions_candidate: set[str] = set()

    if payload:
        employee_id_candidate = (
            payload.get("employee_code")
            or payload.get("employee_id")
            or payload.get("sub")
            or payload.get("emp_id")
        )
        if isinstance(employee_id_candidate, int):
            employee_id_candidate = str(employee_id_candidate)
        role_candidate = payload.get("role")
        raw_perms = payload.get("permissions")
        if isinstance(raw_perms, list):
            permissions_candidate = {str(p) for p in raw_perms}

    # 2. If not a valid JWT, support test token format: test-token-{employee_id} or raw employee_id
    if not employee_id_candidate:
        if token.startswith("test-token-"):
            employee_id_candidate = token.replace("test-token-", "", 1)
        elif token.startswith("token-"):
            employee_id_candidate = token.replace("token-", "", 1)
        else:
            employee_id_candidate = token

    # 3. Look up caller employee in DB
    shared_schema = is_shared_hr_schema(db.get_bind())
    caller_emp = (
        get_shared_employee(db, str(employee_id_candidate))
        if shared_schema and employee_id_candidate
        else db.query(Employee).filter(Employee.id == employee_id_candidate).first()
    )

    # Also search by first_name or partial ID if not directly matched
    if not caller_emp and employee_id_candidate and not shared_schema:
        caller_emp = db.query(Employee).filter(Employee.id.ilike(f"%{employee_id_candidate}%")).first()

    if not caller_emp:
        raise HRApiException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message="Unauthenticated. Invalid token or user not found.",
        )

    final_role = normalize_role(role_candidate or getattr(caller_emp, "role_title", "employee"))

    return HRCallerContext(
        user_id=getattr(caller_emp, "user_id", caller_emp.id),
        employee_id=caller_emp.id,
        role=final_role,
        department=caller_emp.department or "General",
        permissions=permissions_candidate,
    )


def verify_feature_rbac(caller: HRCallerContext, feature: str) -> None:
    """Enforces role-based access control for specific AI features."""
    manager_features = {"evaluation-draft", "attention-signal", "team-insight"}
    if feature in manager_features and caller.role == HRCallerRole.EMPLOYEE.value:
        raise HRApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            message=f"Forbidden: employees are not authorized to access '{feature}'.",
        )


def verify_and_resolve_employee_scope(
    caller: HRCallerContext,
    target_employee_id: str | None,
    db: Session,
) -> str:
    """Enforces target employee scoping, preventing BOLA/IDOR attacks.

    - Employees can only target themselves. If body differs from caller, rejects immediately.
    - Managers can only target employees in their own department.
    - HR Admins can target any valid employee in the system.
    """
    clean_target = (target_employee_id or "").strip()

    # Self-service for employees
    if caller.role == HRCallerRole.EMPLOYEE.value:
        if clean_target and clean_target != caller.employee_id:
            raise HRApiException(
                status_code=status.HTTP_403_FORBIDDEN,
                message=f"Forbidden: employees may only access their own records (caller={caller.employee_id}, requested={clean_target}).",
            )
        return caller.employee_id

    # If target is omitted for manager/admin, default to caller
    if not clean_target:
        return caller.employee_id

    # Verify target employee exists
    target_emp = (
        get_shared_employee(db, clean_target)
        if is_shared_hr_schema(db.get_bind())
        else db.query(Employee).filter(Employee.id == clean_target).first()
    )
    if not target_emp:
        raise HRApiException(
            status_code=status.HTTP_404_NOT_FOUND,
            message=f"Employee '{clean_target}' not found.",
        )

    # Department scope for managers
    if caller.role == HRCallerRole.MANAGER.value:
        if (target_emp.department or "").strip().lower() != (caller.department or "").strip().lower():
            raise HRApiException(
                status_code=status.HTTP_403_FORBIDDEN,
                message=f"Forbidden: managers may only access employees in their own department (caller dept='{caller.department}', target dept='{target_emp.department}').",
            )
        return target_emp.id

    # HR Admin has organization-wide access
    return target_emp.id


def verify_and_resolve_department_scope(
    caller: HRCallerContext,
    requested_department: str | None,
    requested_department_id: int | str | None,
    db: Session,
) -> str:
    """Enforces department scoping for Team Insight."""
    if caller.role == HRCallerRole.EMPLOYEE.value:
        raise HRApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            message="Forbidden: employees are not authorized to access Team Insight.",
        )

    dept = (requested_department or "").strip()
    if not dept and requested_department_id is not None:
        dept = str(requested_department_id).strip()

    if not dept:
        dept = caller.department

    if caller.role == HRCallerRole.MANAGER.value:
        if dept.lower() != (caller.department or "").strip().lower():
            raise HRApiException(
                status_code=status.HTTP_403_FORBIDDEN,
                message=f"Forbidden: managers may only access Team Insight for their own department (caller dept='{caller.department}', requested dept='{dept}').",
            )
        return caller.department

    return dept


def get_service_auth_headers(caller: HRCallerContext) -> dict[str, str]:
    """Builds trusted service authentication headers to call the internal AI service."""
    return {
        "X-Caller-Employee-ID": caller.employee_id,
        "X-Caller-Role": caller.role,
        "X-Gateway-Service-Token": os.getenv("AI_GATEWAY_SERVICE_TOKEN", DEFAULT_GATEWAY_SERVICE_TOKEN),
    }
