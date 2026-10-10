"""HR Backend Gateway module.

Provides authentication, RBAC, scope validation, internal service dispatch,
response mapping, and error translation for AI operations.
"""

from app.gateway.routes import router as hr_gateway_router
from app.gateway.security import HRApiException

__all__ = ["HRApiException", "hr_gateway_router"]
