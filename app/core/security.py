"""Authentication and authorization dependencies for trusted gateway caller context.

Implements caller identity verification from trusted gateway headers (X-Caller-Employee-ID,
X-Caller-Role), immutable caller context generation, and strict scope authorization
across employee-scoped, manager-only, and department-scoped AI routes.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import ChatSession, Employee
from app.services.shared_hr_data import get_shared_employee, is_shared_hr_schema


class CallerRole(str, Enum):
    """Allowed authorization roles passed by trusted gateway."""

    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR_ADMIN = "hr_admin"


ALLOWED_ROLES = {CallerRole.EMPLOYEE.value, CallerRole.MANAGER.value, CallerRole.HR_ADMIN.value}

# These headers are populated by the HR gateway.  Declaring the identity header
# as an API-key security scheme makes the trust boundary visible in OpenAPI
# while preserving the existing runtime contract used by the gateway.
caller_identity_header = APIKeyHeader(
    name="X-Caller-Employee-ID",
    scheme_name="GatewayCallerIdentity",
    auto_error=False,
    description="Trusted caller identity set by the HR gateway; clients must not set this directly.",
)


@dataclass(frozen=True)
class CallerContext:
    """Immutable caller identity and scope passed by the trusted gateway."""

    employee_id: str
    role: str
    department: str


def get_caller_context(
    x_caller_employee_id: Annotated[str | None, Security(caller_identity_header)] = None,
    x_caller_role: Annotated[
        str | None,
        Header(
            alias="X-Caller-Role",
            description="Authenticated caller role (employee, manager, hr_admin)",
        ),
    ] = None,
    x_gateway_service_token: Annotated[
        str | None,
        Header(
            alias="X-Gateway-Service-Token",
            description="Service credential required when AI_GATEWAY_SERVICE_TOKEN is configured",
        ),
    ] = None,
    db: Annotated[Session, Depends(get_db)] = None,
) -> CallerContext:
    """Extracts and verifies authenticated caller context from trusted gateway headers.

    Rules:
    - Requires X-Caller-Employee-ID (non-empty).
    - Requires X-Caller-Role (non-empty).
    - Validates caller exists in the Employee database table.
    - Validates role is one of: employee, manager, hr_admin.
    - Returns an immutable CallerContext(employee_id, role, department).
    - Missing or invalid caller headers -> HTTP 401 Unauthorized.
    """
    configured_token = os.getenv("AI_GATEWAY_SERVICE_TOKEN", "").strip()
    if configured_token and not hmac.compare_digest(x_gateway_service_token or "", configured_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing HR gateway service credential",
            headers={"WWW-Authenticate": "Gateway"},
        )
    if not x_caller_employee_id or not x_caller_employee_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or empty required header: X-Caller-Employee-ID",
            headers={"WWW-Authenticate": "Gateway"},
        )

    if not x_caller_role or not x_caller_role.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or empty required header: X-Caller-Role",
        )

    clean_role = x_caller_role.strip().lower()
    if clean_role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid caller role '{x_caller_role}'. Allowed roles: employee, manager, hr_admin",
        )

    clean_employee_id = x_caller_employee_id.strip()
    caller_emp = (
        get_shared_employee(db, clean_employee_id)
        if is_shared_hr_schema(db.get_bind())
        else db.query(Employee).filter(Employee.id == clean_employee_id).first()
    )
    if not caller_emp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid caller employee ID '{clean_employee_id}'. Caller does not exist.",
        )

    return CallerContext(
        employee_id=caller_emp.id,
        role=clean_role,
        department=caller_emp.department,
    )


def authorize_employee_scope(
    caller: CallerContext,
    target_employee_id: str,
    db: Session,
    is_manager_only: bool = False,
) -> str:
    """Enforces authorization scope for employee-targeted AI endpoints.

    Rules:
    - If is_manager_only is True, employee role is rejected with HTTP 403.
    - Employee role can only access their own records (target_employee_id == caller.employee_id).
    - Manager role can only access employees within their own department.
    - HR Admin role can access any employee.

    Returns:
        The authorized target employee_id string.
    """
    clean_target_id = target_employee_id.strip() if target_employee_id else ""

    # 1. Manager-only features: Employees cannot access
    if is_manager_only and caller.role == CallerRole.EMPLOYEE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: employees are not authorized to access manager-level AI features.",
        )

    # 2. Employee role: self-service only
    if caller.role == CallerRole.EMPLOYEE.value:
        if clean_target_id != caller.employee_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: employees may only access their own records (caller={caller.employee_id}, requested={clean_target_id}).",
            )
        return caller.employee_id

    # 3. Manager role: department-scoped
    if caller.role == CallerRole.MANAGER.value:
        target_emp = (
            get_shared_employee(db, clean_target_id)
            if is_shared_hr_schema(db.get_bind())
            else db.query(Employee).filter(Employee.id == clean_target_id).first()
        )
        if target_emp and target_emp.department.strip().lower() != caller.department.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: managers may only access employees in their own department (caller dept='{caller.department}', target dept='{target_emp.department}').",
            )
        return clean_target_id

    # 4. HR Admin role: organization-wide access
    if caller.role == CallerRole.HR_ADMIN.value:
        return clean_target_id

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Forbidden: unauthorized caller role.",
    )


def authorize_department_scope(
    caller: CallerContext,
    requested_department: str,
) -> str:
    """Enforces authorization scope for department-targeted AI endpoints (e.g. Team Insight).

    Rules:
    - Employee role cannot access Team Insight (HTTP 403).
    - Manager role can only access Team Insight for their own department (HTTP 403 if mismatch).
    - HR Admin role can access any department.

    Returns:
        The authorized department string.
    """
    clean_dept = requested_department.strip() if requested_department else ""

    # 1. Employees cannot access Team Insight
    if caller.role == CallerRole.EMPLOYEE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: employees are not authorized to access Team Insight.",
        )

    # 2. Managers may only access their own department
    if caller.role == CallerRole.MANAGER.value:
        if clean_dept.lower() != caller.department.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: managers may only access Team Insight for their own department (caller dept='{caller.department}', requested dept='{clean_dept}').",
            )
        return caller.department

    # 3. HR Admin has organization-wide access
    if caller.role == CallerRole.HR_ADMIN.value:
        return clean_dept

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Forbidden: unauthorized caller role.",
    )


def authorize_chat_session_scope(
    caller: CallerContext,
    session_id: str | None,
    target_employee_id: str,
    db: Session,
) -> None:
    """Enforces chat session ownership against caller context in Policy Assistant.

    Rules:
    - Session must belong to target_employee_id.
    - Employee callers can only access sessions belonging to themselves.
    - Manager callers can only access sessions belonging to employees in their department.
    """
    if not session_id or not str(session_id).strip():
        return

    clean_session_id = str(session_id).strip()
    session = db.query(ChatSession).filter(ChatSession.id == clean_session_id).first()
    if not session:
        # Session not found will be handled by the service (HTTP 404)
        return

    if session.employee_id != target_employee_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: chat session belongs to another employee.",
        )

    if caller.role == CallerRole.EMPLOYEE.value and session.employee_id != caller.employee_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: chat session belongs to another employee.",
        )

    if caller.role == CallerRole.MANAGER.value:
        session_emp = (
            get_shared_employee(db, session.employee_id)
            if is_shared_hr_schema(db.get_bind())
            else db.query(Employee).filter(Employee.id == session.employee_id).first()
        )
        if session_emp and session_emp.department.strip().lower() != caller.department.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: chat session belongs to an employee outside your department.",
            )
