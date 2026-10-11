import secrets
import typing as t
import re
import httpx
from urllib.parse import urlencode
from fastapi import Request, Depends, status as http_status
from fastapi.responses import RedirectResponse
from dsp_decikit.settings import settings
from dsp_decikit.httpx_client import httpx_client_factory
from dsp_decikit.auth.sessions import put_login_state, take_login_state
from dsp_decikit.exceptions import UnauthorizedException
from .jwks import get_jwks_manager, JWKSManager
from dsp_decikit.auth.authorization import (
    update_user_and_get_session,
    update_user_and_get_session_with_team_roles,
)
from dsp_decikit.auth.models import UserInfo
from dsp_decikit.auth.models import TeamRole
from dsp_decikit.auth.db_service import find_user_by_email, get_user_teams
from fastapi import APIRouter
from logging import getLogger
import time

if t.TYPE_CHECKING:
    from dsp_decikit.auth.settings import _OidcAuthSettings

logger = getLogger(__name__)
router = APIRouter()

# Not a fan of using this globally but it seems to be the cleanest
oidc_settings = settings.auth.oidc


class LoginSessionState(t.TypedDict):
    state: str
    provider: str


def apply_group_remapping(
    groups: t.List[str], remapping_rules: t.List
) -> t.List[t.Tuple[str, TeamRole]]:
    """Apply group remapping rules to convert OIDC groups to internal team names with roles"""
    mapped_team_roles = []

    for group in groups:
        mapped = False
        for rule in remapping_rules:
            try:
                # Apply regex find/replace
                if re.search(rule.find, group):
                    mapped_team = re.sub(rule.find, rule.replace, group)
                    mapped_team_roles.append((mapped_team, rule.default_role))
                    mapped = True
                    break
            except re.error:
                # Skip invalid regex patterns
                continue

    return mapped_team_roles


def _build_redirect_uri(request: Request, oidc_settings: "_OidcAuthSettings") -> str:
    """
    Build the OAuth redirect URI, handling reverse proxy/ingress scenarios.

    Uses configuration to override base URL or force HTTPS when behind a proxy.
    Supports format strings like: https://example.com{redirect_path}
    """
    redirect_path = request.app.url_path_for("auth_oidc_redirect")

    # If redirect_url_override is provided, use it with format string support
    if oidc_settings.redirect_url_override:
        return oidc_settings.redirect_url_override.format(redirect_path=redirect_path)

    # Otherwise, use request.base_url but potentially force HTTPS
    base_url = request.base_url.replace(path="")

    if oidc_settings.force_https_redirects and base_url.scheme == "http":
        base_url = base_url.replace(scheme="https")

    return str(base_url.replace(path=redirect_path))


@router.get(oidc_settings.login_path)
async def auth_oidc_login(
    provider: str,
    request: Request,
    jwks_manager: JWKSManager = Depends(get_jwks_manager),
) -> RedirectResponse:
    try:
        # Validate provider exists
        if provider not in oidc_settings.providers:
            raise UnauthorizedException(f"Provider '{provider}' not configured")

        provider_config = oidc_settings.providers[provider]
        session_id = secrets.token_urlsafe(oidc_settings.session_token_length)
        state = secrets.token_urlsafe(oidc_settings.session_token_length)

        params = {
            "client_id": provider_config.client_id,
            "scope": provider_config.scopes,
            "response_type": "code",
            "redirect_uri": _build_redirect_uri(request, oidc_settings),
            "state": state,
            "response_mode": "query",
        }
        oidc_config = await jwks_manager.get_oidc_config(provider)
        auth_url = f"{oidc_config.authorization_endpoint}?{urlencode(params)}"

        await put_login_state(
            session_id,
            LoginSessionState(state=state, provider=provider),
            oidc_settings.login_session_timeout_s,
        )
        response = RedirectResponse(url=auth_url)
        response.set_cookie(
            value=session_id,
            max_age=oidc_settings.login_session_timeout_s,
            **oidc_settings.cookie.model_dump(mode="json", exclude={"key"}),
            key=oidc_settings.cookie.key,
        )
        return response
    except Exception as e:
        logger.exception(e)
        raise UnauthorizedException(
            f"An internal failure occurred when signing you in."
        )


@router.get(oidc_settings.redirect_path, name="auth_oidc_redirect")
async def auth_oidc_redirect(
    request: Request,
    code: str,
    state: str,
    jwks_manager: JWKSManager = Depends(get_jwks_manager),
) -> str:
    login_session_id = request.cookies.get(oidc_settings.cookie.key)
    login_session_info = login_session_id and await take_login_state(login_session_id)
    if not login_session_info:
        raise UnauthorizedException("Invalid session")
    session_state = login_session_info.get("state")
    provider = login_session_info.get("provider")

    # We need to also just make sure there is no attacks with state=None
    if not session_state or session_state != state:
        raise UnauthorizedException("Invalid state parameter")

    # Validate provider from session
    if not provider or provider not in oidc_settings.providers:
        raise UnauthorizedException("Invalid provider in session")

    provider_config = oidc_settings.providers[provider]
    oidc_config = await jwks_manager.get_oidc_config(provider)

    async with httpx_client_factory() as client:
        response = await client.post(
            oidc_config.token_endpoint,
            data={
                "client_id": provider_config.client_id,
                "client_secret": provider_config.client_secret.get_secret_value(),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": _build_redirect_uri(request, oidc_settings),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.exception(e)
            try:
                logger.error(
                    f"Response info:\n status:{e.response.status_code}\n body:{e.response.read()}"
                )
            except Exception as e:
                pass
            raise UnauthorizedException("Could not retrieve token from server")
    token_response = response.json()
    id_token = token_response.get("id_token")
    if not id_token:
        raise UnauthorizedException("No ID token received")

    # Verify JWT token signature using jwks_manager
    try:
        decoded_payload = await jwks_manager.verify_jwt(id_token, provider)
    except ValueError as e:
        raise UnauthorizedException(str(e))

    # Extract user information from token
    user_info = UserInfo(
        name=decoded_payload.get("name", ""), email=decoded_payload.get("email", "")
    )

    # Extract groups from token using provider-specific groups field
    groups = decoded_payload.get(provider_config.groups_field, [])

    # Apply OIDC-specific group remapping rules with roles
    if len(provider_config.group_remapping_rules):
        mapped_team_roles = apply_group_remapping(
            groups, provider_config.group_remapping_rules
        )
    else:
        mapped_team_roles = groups
    if not len(mapped_team_roles):
        raise UnauthorizedException(
            "The user did not belong to any groups or the users groups did not match any pattern in the group remapping rules."
        )

    # Implement group preservation - merge with existing user groups instead of replacing
    existing_user = await find_user_by_email(user_info.email)
    if existing_user:
        # Get existing teams for the user
        existing_teams = await get_user_teams(existing_user.id)
        # Convert existing teams to team_roles with MEMBER role (preserving existing memberships)
        existing_team_roles = [
            (team_name, TeamRole.MEMBER) for team_name in existing_teams
        ]

        # Merge existing teams with new OAuth teams, avoiding duplicates
        # New OAuth teams take precedence for role assignment
        oauth_team_names = {team_name for team_name, role in mapped_team_roles}
        preserved_teams = [
            (team_name, role)
            for team_name, role in existing_team_roles
            if team_name not in oauth_team_names
        ]

        # Combine preserved teams with new OAuth teams
        all_team_roles = preserved_teams + mapped_team_roles
    else:
        # New user - just use OAuth teams
        all_team_roles = mapped_team_roles

    session_id, _ = await update_user_and_get_session_with_team_roles(
        user_info, team_roles=all_team_roles
    )

    # Handle redirect URL from login state
    # For now, default to root - could be enhanced to store original URL in login state
    redirect_url = "/"

    response = RedirectResponse(url=redirect_url)
    response.delete_cookie(
        **oidc_settings.cookie.model_dump(mode="json", exclude={"key"}),
        key=oidc_settings.cookie.key,
    )

    # Set main session cookie (httponly, secure)
    response.set_cookie(
        key=settings.auth.cookie.key,
        value=session_id,
        max_age=settings.auth.session_max_duration_s,  # Use max duration for cookie expiry
        **settings.auth.cookie.model_dump(mode="json", exclude={"key"}),
    )

    # Set session status cookie (JS readable, contains expiry timestamp)
    # Calculate expiry from session creation time + rolling duration
    current_time = int(time.time())
    expiry_timestamp = current_time + settings.auth.session_rolling_duration_s
    response.set_cookie(
        key=settings.auth.status_cookie.key,
        value=str(expiry_timestamp),
        max_age=settings.auth.session_max_duration_s,  # Use max duration for cookie expiry
        **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
    )

    return response
