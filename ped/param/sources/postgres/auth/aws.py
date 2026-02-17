"""_summary_

Note this package isnt registerd by default to the set of postgres auth provider.
Therefore to use this package you must first
from ped.param.sources.postgres.auth.aws import AWSAuth
This will register the AWSAuth provider and make it available for use in your database settings.

Returns:
    _type_: _description_
"""

import typing as t
import time
import threading
from pydantic import Field, PrivateAttr
from .core import BaseAuth
from ._ext import register_auth_provider

if t.TYPE_CHECKING:
    from ..core import DatabaseSettings


class AWSAuth(BaseAuth):
    """AWS IAM authentication for RDS"""
    method: t.Literal["aws"] = "aws"
    username: str = "postgres"
    region: str = Field(description="AWS region for RDS instance")
    
    # Token expiration settings
    token_duration_minutes: int = Field(default=15, description="Token validity duration in minutes")
    refresh_buffer_minutes: int = Field(default=5, description="Refresh token this many minutes before expiration")
    
    # Optional role assumption
    role_arn: t.Optional[str] = Field(default=None, description="IAM role ARN to assume before generating token")
    
    # Private attributes for token caching
    _auth_token: t.Optional[str] = PrivateAttr(default=None)
    _token_expires_at: float = PrivateAttr(default=0)
    _token_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_boto_session(self):
        """Get boto3 session, optionally with role assumption"""
        import boto3

        if self.role_arn:
            # Assume role first
            sts = boto3.client("sts")
            response = sts.assume_role(
                RoleArn=self.role_arn, 
                RoleSessionName="ped-postgres-access"
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

    def _generate_auth_token(self, db_settings: "DatabaseSettings") -> str:
        """Generate RDS IAM authentication token"""
        session = self._get_boto_session()
        rds = session.client("rds")

        return rds.generate_db_auth_token(
            DBHostname=db_settings.host,
            Port=db_settings.port,
            DBUsername=self.username,
        )

    def _get_current_auth_token(self, db_settings: "DatabaseSettings", force_refresh: bool = False) -> str:
        """Get current auth token, refreshing if needed (thread-safe)"""
        now = time.time()
        buffer_seconds = self.refresh_buffer_minutes * 60

        # Fast path - check without lock first
        if (
            not force_refresh
            and self._auth_token
            and now < (self._token_expires_at - buffer_seconds)
        ):
            return self._auth_token

        # Slow path - acquire lock and refresh if still needed
        with self._token_lock:
            # Double-check after acquiring lock
            if (
                not force_refresh
                and self._auth_token
                and now < (self._token_expires_at - buffer_seconds)
            ):
                return self._auth_token

            # Generate new token
            self._auth_token = self._generate_auth_token(db_settings)
            self._token_expires_at = now + (self.token_duration_minutes * 60)
            return self._auth_token

    def invalidate_auth(self):
        """Invalidate current auth token to force regeneration on next use"""
        with self._token_lock:
            self._auth_token = None
            self._token_expires_at = 0

    def get_connection_params(self, db_settings: "DatabaseSettings") -> t.Dict[str, t.Any]:
        """Get connection parameters for psycopg3"""
        auth_token = self._get_current_auth_token(db_settings)
        
        return {
            "host": db_settings.host,
            "port": db_settings.port,
            "dbname": db_settings.database,
            "user": self.username,
            "password": auth_token,
            "sslmode": "require",  # AWS RDS requires SSL
        }


register_auth_provider(AWSAuth)