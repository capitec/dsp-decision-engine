import typing as t
import time
import threading
from urllib.parse import urlencode, urlunparse, urlparse, parse_qs, quote
from pydantic import BaseModel, Field, SecretStr, Tag, Discriminator, PrivateAttr
from collections import namedtuple

# Named tuple for URL components
UrlParts = namedtuple(
    "UrlParts", ["scheme", "netloc", "path", "params", "query", "fragment"]
)


class BasicAuth(BaseModel):
    method: t.Literal["basic"] = "basic"
    username: str = "postgres"
    password: SecretStr = SecretStr("postgres")

    def add_auth(
        self, url_parts: UrlParts, db_settings: "DatabaseSettings"
    ) -> UrlParts:
        """Add basic auth credentials to URL parts"""
        return url_parts._replace(
            netloc=f"{self.username}:{self.password.get_secret_value()}@{url_parts.netloc}"
        )

    def event_do_connect_hook(self, cparams: dict, db_settings: "DatabaseSettings"):
        pass


class AWSAuth(BaseModel):
    method: t.Literal["aws"] = "aws"
    username: str = "postgres"
    region: str = Field(description="AWS region for RDS instance")

    # Token expiration settings
    token_duration_minutes: int = Field(
        default=15, description="Token validity duration in minutes"
    )
    refresh_buffer_minutes: int = Field(
        default=5, description="Refresh token this many minutes before expiration"
    )

    # Optional role assumption
    role_arn: t.Optional[str] = Field(
        default=None, description="IAM role ARN to assume before generating token"
    )

    # Auth parameters cache (class variables to persist across instances)
    _auth_token: t.Optional[str] = None
    _params_expires_at: float = 0
    _token_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_boto_session(self):
        """Get boto3 session, optionally with role assumption"""
        import boto3

        if self.role_arn:
            # Assume role first
            sts = boto3.client("sts")
            response = sts.assume_role(
                RoleArn=self.role_arn, RoleSessionName="dsp-decider-db-access"
            )
            credentials = response["Credentials"]

            return boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
                region_name=self.region,
            )
        else:
            return boto3.Session(region_name=self.region)

    def generate_signed_url(self, db_settings: "DatabaseSettings") -> str:
        """Generate RDS IAM authentication signed URL"""
        session = self._get_boto_session()
        rds = session.client("rds")

        signed_url = rds.generate_db_auth_token(
            DBHostname=db_settings.host,
            Port=db_settings.port,
            DBUsername=self.username,
        )
        # This generates a presigned url like:
        # host:port/?Action=connect&DBUser=...&X-Amz-Algorithm=AWS4-HMAC-SHA256
        # &X-Amz-Credential=...&X-Amz-Date=20250811T103601Z&X-Amz-Expires=900
        # &X-Amz-SignedHeaders=host&X-Amz-Security-Token=...&X-Amz-Signature=...
        return signed_url

    def _parse_signed_url_params(self, signed_url: str) -> t.Dict[str, str]:
        """Parse AWS signed URL to extract auth parameters"""
        parsed = urlparse(signed_url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        # Flatten single-value lists from parse_qs
        flattened_params = {}
        for key, values in params.items():
            if key in ("Action",):
                continue
            flattened_params[key] = values[0] if values else ""

        return flattened_params

    def _extract_expiration_from_params(self, params: t.Dict[str, str]) -> float:
        """Extract expiration timestamp from AWS auth parameters"""
        # Try to get expiration from X-Amz-Expires (duration in seconds)
        expires_duration = params["X-Amz-Expires"]
        return time.time() + int(expires_duration)

    def get_current_auth_params(
        self, db_settings: "DatabaseSettings", force_refresh: bool = False
    ) -> str:
        """Get current auth token, refreshing if needed or forced (thread-safe)"""
        now = time.time()
        buffer_seconds = self.refresh_buffer_minutes * 60

        # Fast path - check without lock first
        if (
            not force_refresh
            and self._auth_token
            and now < (self._params_expires_at - buffer_seconds)
        ):
            return self._auth_token

        # Slow path - acquire lock and refresh if still needed
        with self._token_lock:
            # Double-check after acquiring lock (another thread might have refreshed)
            if (
                not force_refresh
                and self._auth_token
                and now < (self._params_expires_at - buffer_seconds)
            ):
                return self._auth_token

            # Generate new token
            _auth_token = self.generate_signed_url(db_settings)
            try:
                _auth_params = self._parse_signed_url_params(_auth_token)
                self._params_expires_at = self._extract_expiration_from_params(
                    _auth_params
                )
            except Exception:
                # If parsing fails, default to configured duration
                self._params_expires_at = now + (self.token_duration_minutes * 60)

            self._auth_token = _auth_token
            return self._auth_token

    def invalidate_auth_params(self):
        """Invalidate current auth parameters to force regeneration on next use"""
        with self._token_lock:
            self._auth_token = None
            self._params_expires_at = 0

    def add_auth(
        self, url_parts: UrlParts, db_settings: "DatabaseSettings"
    ) -> UrlParts:
        """Add AWS username to URL parts (password handled via event hook)"""
        # password = self.get_current_auth_params(db_settings)
        return url_parts._replace(netloc=f"{self.username}@{url_parts.netloc}")

    def event_do_connect_hook(self, cparams: dict, db_settings: "DatabaseSettings"):
        # For AWS, we'll need to handle the signed URL parameters differently
        # This will be updated when the hook implementation is sorted out
        auth_params = self.get_current_auth_params(db_settings)
        # TODO: Update when event__do_connect hook is implemented
        # print(f"AWS auth params available: {list(auth_params.keys())}")
        # cparams.update(**auth_params)
        cparams["password"] = auth_params


def _db_auth_discriminator(v):
    if isinstance(v, dict):
        # Default to "basic" if method not specified
        return v.get("method", "basic")
    return getattr(v, "method", "basic")


_TDBAuth = t.Annotated[
    t.Union[t.Annotated[BasicAuth, Tag("basic")], t.Annotated[AWSAuth, Tag("aws")]],
    Discriminator(_db_auth_discriminator),
]


class DatabaseSettings(BaseModel):
    """Database connection and configuration settings"""

    # Connection settings
    host: str = "localhost"
    port: int = 5432
    auth: _TDBAuth = Field(default_factory=BasicAuth)
    database: str = "postgres"
    pg_schema: str = Field(default="public", alias="schema")

    # Driver settings
    async_driver: str = Field(
        default="postgresql+asyncpg", description="Async database driver"
    )

    # Connection pool settings
    max_pool_size: int = Field(
        default=20, description="Maximum number of connections in the pool"
    )
    max_overflow: int = Field(
        default=10, description="Maximum number of connections to overflow"
    )
    pool_timeout: int = Field(
        default=30, description="Timeout in seconds for getting connection from pool"
    )
    pool_recycle: int = Field(
        default=600,
        description="Recycle connections after this many seconds (10 min for AWS token refresh)",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Test connections before use to handle dropped connections",
    )
    pool_reset_on_return: str = Field(
        default="commit",
        description="Reset connections on return (commit/rollback/none)",
    )

    # SQLAlchemy settings
    echo: bool = Field(default=False, description="Enable SQL query logging")
    echo_pool: bool = Field(default=False, description="Enable connection pool logging")

    # URL parameters - flexible key-value pairs for any connection parameters
    url_params: t.Dict[str, str] = Field(
        default_factory=lambda: {},
        description="Additional URL parameters as key-value pairs",
    )

    # Connection arguments - driver-specific connection parameters
    connect_args: t.Dict[str, t.Any] = Field(
        default_factory=lambda: {
            "server_settings": {
                "application_name": "dsp-decider",
                "tcp_keepalives_idle": "600",  # 10 minutes
                "tcp_keepalives_interval": "30",  # 30 seconds
                "tcp_keepalives_count": "3",  # 3 attempts
            },
            # SSL configuration for AWS RDS
            "ssl": "allow",  # require, prefer, allow, disable
        },
        description="Driver-specific connection arguments (overrides defaults)",
    )

    def get_execution_options(self) -> t.Dict[str, t.Any]:
        """Get execution options with schema translation map"""
        return {
            "schema_translate_map": {
                "app_schema": self.pg_schema  # always translate, even if 'public'
            }
        }

    def get_connect_args(self) -> t.Dict[str, t.Any]:
        """Get database-specific connection arguments with defaults and user overrides"""
        # Default connection arguments based on driver
        # defaults = {}
        # if "asyncpg" in self.async_driver:
        #     defaults = {
        #         "server_settings": {
        #             "application_name": "dsp-decider",
        #             "tcp_keepalives_idle": "600",  # 10 minutes
        #             "tcp_keepalives_interval": "30",  # 30 seconds
        #             "tcp_keepalives_count": "3",  # 3 attempts
        #         }
        #     }
        # elif "psycopg" in self.async_driver or "psycopg" in getattr(self, 'sync_driver', ''):
        #     defaults = {
        #         "application_name": "dsp-decider",
        #         "keepalives_idle": 600,  # 10 minutes
        #         "keepalives_interval": 30,  # 30 seconds
        #         "keepalives_count": 3,  # 3 attempts
        #     }

        # # Merge defaults with user-provided connect_args
        # result = defaults.copy()
        # if self.connect_args:
        #     # Deep merge for nested dictionaries like server_settings
        #     for key, value in self.connect_args.items():
        #         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
        #             result[key] = {**result[key], **value}
        #         else:
        #             result[key] = value

        return self.connect_args

    def _build_url(self, driver: str) -> str:
        """Build database URL with proper URL encoding (no schema options)"""
        # Prepare base URL parameters (no schema options - handled by translation map)
        params = self.url_params.copy()

        # Build base URL components
        netloc = f"{self.host}:{self.port}"
        path = f"/{self.database}"
        query = urlencode(params) if params else ""

        # Create base URL parts
        url_parts = UrlParts(driver, netloc, path, "", query, "")

        # Let auth method add authentication
        authenticated_url_parts = self.auth.add_auth(url_parts, self)

        return urlunparse(authenticated_url_parts)

    @property
    def database_url(self) -> str:
        """Build the async database URL"""
        return self._build_url(self.async_driver)
