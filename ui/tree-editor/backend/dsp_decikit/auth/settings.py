import typing as t
import json
from typing import Annotated
from pydantic import Field, BaseModel, SecretStr, BeforeValidator, SecretBytes

if t.TYPE_CHECKING:
    from fastapi import APIRouter

# Import TeamRole and OAuthProvider for role mapping and provider configuration
from dsp_decikit.auth.models import TeamRole, OAuthProvider


def enabled_pre_validator(v: t.Any) -> str:
    if isinstance(v, dict):
        enabled_val = v.get("enabled", False)
        if isinstance(enabled_val, str):
            # Convert string representations to boolean
            if enabled_val.lower() in ("true", "1", "yes", "on"):
                enabled_val = True
            elif enabled_val.lower() in ("false", "0", "no", "off", ""):
                enabled_val = False
        return {**v, "enabled": bool(enabled_val)}
    v.enabled = bool(getattr(v, "enabled", False))
    return v


def parse_group_remapping_rules(v: t.Any) -> t.List[t.Dict[str, t.Any]]:
    """Parse group remapping rules from either a list or JSON string"""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("group_remapping_rules JSON must be a list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in group_remapping_rules: {e}")
    elif isinstance(v, list):
        return v
    elif v is None:
        return []
    else:
        raise ValueError("group_remapping_rules must be a list, JSON string, or None")


class TeamRoleMapping(BaseModel):
    team_name: str
    role: TeamRole = TeamRole.MEMBER


class GroupRemappingRule(BaseModel):
    find: str  # regex pattern to find
    replace: str  # replacement pattern
    default_role: TeamRole = TeamRole.MEMBER  # default role for mapped groups


class SupportsInit(t.Protocol):
    def init_provider(self, auth_router: "APIRouter") -> None: ...


class _DisabledSettings(BaseModel):
    enabled: t.Literal[False] = False

    def init_provider(self, auth_router: "APIRouter"):
        pass


class _OidcProviderSettings(BaseModel):
    client_id: str
    client_secret: SecretStr
    authority: str
    scopes: str = "openid profile email groups"

    # OAuth provider type - determines branding and icons
    oauth_provider: OAuthProvider = OAuthProvider.OTHER

    # UI branding for this OIDC provider (only used when oauth_provider is OTHER)
    display_name: t.Optional[str] = None
    icon_name: t.Optional[str] = None

    # OIDC-specific group remapping rules (moved from global settings)
    group_remapping_rules: Annotated[
        t.List[GroupRemappingRule], BeforeValidator(parse_group_remapping_rules)
    ] = Field(default_factory=list)
    groups_field: str = "groups"

    @property
    def oidc_well_known_url(self):
        return f"{self.authority}/.well-known/openid-configuration"

    def get_display_name(self) -> str:
        """Get display name based on OAuth provider or custom setting"""
        if self.display_name:
            return self.display_name
        if self.oauth_provider == OAuthProvider.MICROSOFT:
            return "Microsoft"
        elif self.oauth_provider == OAuthProvider.GITHUB:
            return "GitHub"
        elif self.oauth_provider == OAuthProvider.GOOGLE:
            return "Google"
        elif self.oauth_provider == OAuthProvider.APPLE:
            return "Apple"
        else:  # OTHER
            return "Single Sign On"

    def get_icon_name(self) -> str:
        """Get icon name based on OAuth provider or custom setting"""
        if self.icon_name:
            return self.icon_name
        if self.oauth_provider == OAuthProvider.MICROSOFT:
            return "microsoft"
        elif self.oauth_provider == OAuthProvider.GITHUB:
            return "github"
        elif self.oauth_provider == OAuthProvider.GOOGLE:
            return "google"
        elif self.oauth_provider == OAuthProvider.APPLE:
            return "apple"
        else:  # OTHER
            return "key"


class CookieConfig(BaseModel):
    key: str = "session_id"
    path: str | None = "/"
    domain: str | None = None
    secure: bool = True
    httponly: bool = True
    samesite: t.Literal["lax", "strict", "none"] | None = "strict"


# Defaults live on subclasses, not in default factories: an env override of one field
# (DSP_AUTH__STATUS_COOKIE__SECURE) rebuilds the model from its class defaults.
class _OidcCookieConfig(CookieConfig):
    key: str = "oidc_session_id"
    samesite: t.Literal["lax", "strict", "none"] | None = "lax"  # OAuth redirects come from the provider


class _StatusCookieConfig(CookieConfig):
    key: str = "session_status"
    httponly: bool = False  # JS reads the expiry


class _OidcAuthSettings(BaseModel):
    enabled: t.Literal[True] = True
    providers: t.Dict[str, _OidcProviderSettings]
    # There is no need to check the jwks endpoint less than every hour
    # We only recheck if there is a kid we dont know about
    jwks_min_refresh_interval_s: int = 3600
    router_prefix: str = "/oidc"
    login_path: str = "/login/{provider}"
    redirect_path: str = "/token"

    # OIDC-specific cookie settings (for OAuth flow only)
    cookie: _OidcCookieConfig = Field(default_factory=_OidcCookieConfig)

    # Reverse proxy/ingress settings
    redirect_url_override: t.Optional[str] = Field(
        default=None,
        description="Override base URL for OAuth redirects with format string support (e.g., 'https://app.example.com{redirect_path}')",
    )
    force_https_redirects: bool = Field(
        default=False,
        description="Force HTTPS scheme in OAuth redirect URLs when behind a proxy/ingress",
    )

    # Session timeout configuration
    login_session_timeout_s: int = 3600  # 1 hour for user to complete OAuth flow
    session_token_length: int = 32  # Length of session tokens

    # If possible use t.TYPE_CHECKING for the api router
    def init_provider(self, auth_router: "APIRouter"):
        from .providers.oidc import router as oidc_router

        auth_router.include_router(
            oidc_router, prefix=self.router_prefix, tags=["oidc"]
        )


_TOidcProviderSettings = t.Annotated[
    t.Union[_DisabledSettings, _OidcAuthSettings],
    BeforeValidator(enabled_pre_validator),
    Field(discriminator="enabled"),
]


class _ApiKeyAuthorizerSettings(BaseModel):
    TOKEN_HASH_LENGTH: t.ClassVar[int] = 32
    PREVIEW_LENGTH: t.ClassVar[int] = 8

    salt_value: SecretBytes = SecretBytes(b"change me to something random")
    # We dont need this too high as its just for token validation and brute forcing the tokens is not as much of a risk as brute forcing a password
    # Also the higher this value the more chance we have of DDOS attacks
    rounds: int = 256
    token_length: int = 32


class _BasicAuthProviderSettings(BaseModel):
    enabled: t.Literal[True] = True
    registration_enabled: bool = True
    email_whitelist: t.List[str] = Field(default_factory=list)
    default_teams_by_domain: t.Dict[str, t.List[TeamRoleMapping]] = Field(
        default_factory=dict
    )

    def init_provider(self, auth_router: "APIRouter"):
        from .providers.basic import router as basic_router

        auth_router.include_router(basic_router, prefix="/basic", tags=["basic-auth"])


_TBasicAuthProviderSettings = t.Annotated[
    t.Union[_DisabledSettings, _BasicAuthProviderSettings],
    BeforeValidator(enabled_pre_validator),
    Field(discriminator="enabled"),
]


class AuthenticationSettings(BaseModel):
    oidc: _TOidcProviderSettings = Field(default_factory=_DisabledSettings)
    basic: _TBasicAuthProviderSettings = Field(
        default_factory=_BasicAuthProviderSettings
    )

    # Global auth settings
    auth_prefix: str = "/auth"
    superadmin_teams: t.List[str] = Field(default_factory=list)

    # Main session cookie (secure, httponly)
    cookie: CookieConfig = Field(default_factory=CookieConfig)

    # Session status cookie (JS readable, contains only expiry timestamp)
    status_cookie: _StatusCookieConfig = Field(default_factory=_StatusCookieConfig)
    session_token_len: int = 32
    session_rolling_duration_s: int = (
        3600  # we will keep a session alive for an hour of no inactivity but if there is it will reset the session duration
    )
    session_max_duration_s: int = (
        60 * 60 * 24
    )  # We will stop sessions after 24 hours no matter what
    session_renewal_threshold_s: int = (
        900  # Renew session when within 15 minutes of expiry
    )
    api_key: _ApiKeyAuthorizerSettings = Field(
        default_factory=_ApiKeyAuthorizerSettings
    )

    def init_provider(self, api_router: "APIRouter"):
        from fastapi import APIRouter
        from .routes import router as auth_config_router

        # Create auth sub-router
        auth_router = APIRouter(prefix=self.auth_prefix, tags=["auth"])

        # Add core auth config routes
        auth_router.include_router(auth_config_router)

        # Initialize provider-specific routes
        self.oidc.init_provider(auth_router)
        self.basic.init_provider(auth_router)

        # Register the auth router with the main API router
        api_router.include_router(auth_router)
