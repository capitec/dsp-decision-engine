import typing as t
import time
from pydantic import BaseModel, SecretStr, EmailStr, Field
import datetime
from uuid import UUID, uuid4
from enum import Enum


class TeamRole(str, Enum):
    MEMBER = "member"
    OWNER = "owner"


class OAuthProvider(str, Enum):
    """Well-known OAuth providers with specific branding and configuration"""

    MICROSOFT = "microsoft"
    GITHUB = "github"
    GOOGLE = "google"
    APPLE = "apple"
    OTHER = "other"


class TokenScope(str, Enum):
    READ = "read"  # Can read projects, export configs
    WRITE = "write"  # Can read + write projects, import configs


class SessionInfo(BaseModel):
    created_at: int = Field(
        default=0
    )  # When session was initially created (unix timestamp)
    last_activity: int = Field(default=0)  # When session was last used (unix timestamp)
    user_id: UUID

    @property
    def expires_at(self) -> int:
        """Calculate session expiry - whichever comes first: rolling expiry or max duration."""
        from dsp_decider.settings import settings

        rolling_expiry = self.last_activity + settings.auth.session_rolling_duration_s
        max_duration_expiry = self.created_at + settings.auth.session_max_duration_s
        return min(rolling_expiry, max_duration_expiry)


class BearerTokenSessionInfo(SessionInfo):
    """Session info for API bearer tokens with fixed expiry times."""

    expires_at: int = Field(
        default_factory=lambda: int(time.time()) + 86400 * 365
    )  # Default 1 year


# These can be cleared or used to update the db models if needed
class UserInfo(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    name: str
    email: EmailStr
    created: datetime.datetime = Field(default_factory=datetime.datetime.now)
    password_hash: t.Optional[SecretStr] = None


class RoleInfo(BaseModel):
    role_id: UUID = Field(default_factory=uuid4)
    name: str
    created: datetime.datetime = Field(default_factory=datetime.datetime.now)
