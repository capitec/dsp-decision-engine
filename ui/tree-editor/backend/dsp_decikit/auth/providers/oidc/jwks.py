import typing as t
from pydantic import BaseModel
from asyncio import Lock
from time import monotonic
from functools import lru_cache
from jose import jwt, jwk
from jose.exceptions import JWTError

from dsp_decikit.httpx_client import httpx_client_factory
from dsp_decikit.settings import settings

# Can maybe move this into the function later
oidc_settings = settings.auth.oidc


class OIDCConfiguration(BaseModel):
    token_endpoint: str
    authorization_endpoint: str
    jwks_uri: str
    issuer: str
    end_session_endpoint: str
    userinfo_endpoint: str
    id_token_signing_alg_values_supported: t.Set[str]
    response_types_supported: t.Set[str]
    scopes_supported: t.Set[str]


class JWKSManager:
    def __init__(self):

        self._oidc_config_lock = Lock()
        self._oidc_configs: t.Dict[str, OIDCConfiguration] = {}  # provider -> config

        self._jwks_keys_lock = Lock()
        self._jwks_keys_last_updated: t.Dict[str, float] = {}  # provider -> timestamp
        self._jwks_keys: t.Dict[str, t.Dict[str, dict]] = {}  # provider -> (kid -> key)

    # Get Config
    async def get_oidc_config(
        self, provider: str, force_refresh=False
    ) -> OIDCConfiguration:
        if provider not in oidc_settings.providers:
            raise ValueError(f"Provider '{provider}' not configured")

        provider_config = oidc_settings.providers[provider]

        if provider not in self._oidc_configs or force_refresh:
            async with self._oidc_config_lock:
                if provider in self._oidc_configs and not force_refresh:
                    return self._oidc_configs[provider]
                async with httpx_client_factory() as client:
                    response = await client.get(provider_config.oidc_well_known_url)
                    response.raise_for_status()
                _oidc_config = OIDCConfiguration.model_validate(response.json())
                self._oidc_configs[provider] = _oidc_config
        return self._oidc_configs[provider]

    # Handle refreshing keystore
    def _check_jwks_timestamp(self, provider: str):
        # Note cannot use 0 here because monotonic might start at 0 at the start of the program would take an hour then for the keys to be refreshed initially
        last_updated = self._jwks_keys_last_updated.get(provider, -float("inf"))
        return (monotonic() - last_updated) < oidc_settings.jwks_min_refresh_interval_s

    async def _update_oidc_keys(self, provider: str):
        # We do this check initially to avoid needing to try get a lock if we dont need to
        if self._check_jwks_timestamp(provider):
            return self._jwks_keys.get(provider, {})
        async with self._jwks_keys_lock:
            # We re-do this check in the lock to avoid needing to
            # re-update if multiple calls come in while this is waiting
            if self._check_jwks_timestamp(provider):
                return self._jwks_keys.get(provider, {})
            oidc_config = await self.get_oidc_config(provider)
            async with httpx_client_factory() as client:
                response = await client.get(oidc_config.jwks_uri)
                # If JWKS endpoint fails, could try force refresh of OIDC config
                response.raise_for_status()
            _oidc_key_list = response.json().get("keys", [])
            _oidc_keys = {
                kid: k_data
                for k_data in _oidc_key_list
                if (kid := k_data.get("kid")) is not None
            }
            self._jwks_keys[provider] = _oidc_keys
            self._jwks_keys_last_updated[provider] = monotonic()
            return _oidc_keys

    async def get_key(self, kid: str, provider: str):
        provider_keys = self._jwks_keys.get(provider, {})
        if kid not in provider_keys:
            provider_keys = await self._update_oidc_keys(provider)
        return provider_keys.get(kid, None)

    async def verify_jwt(self, jwt_token: str, provider: str) -> dict:
        """Verify JWT token and return payload using python-jose"""
        try:
            # Get the header to extract the kid
            header = jwt.get_unverified_header(jwt_token)
            kid = header.get("kid")

            if not kid:
                raise ValueError("No 'kid' in JWT header")

            # Get the signing key
            jwk_data = await self.get_key(kid, provider)
            if not jwk_data:
                raise ValueError(f"Key {kid} not found for provider {provider}")

            # Get OIDC config for issuer validation
            oidc_config = await self.get_oidc_config(provider)
            provider_config = oidc_settings.providers[provider]

            # Verify and decode the JWT
            payload = jwt.decode(
                jwt_token,
                jwk_data,
                algorithms=oidc_config.id_token_signing_alg_values_supported,
                audience=provider_config.client_id,
                issuer=oidc_config.issuer,
            )

            return payload

        except JWTError as e:
            raise ValueError(f"JWT verification failed: {e}")
        except Exception as e:
            raise ValueError(f"JWT verification error: {e}")


@lru_cache(maxsize=1)
def get_jwks_manager():
    return JWKSManager()
