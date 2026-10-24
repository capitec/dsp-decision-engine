from sqlalchemy import Column, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlmodel import Field, SQLModel


import typing as t
from datetime import datetime, timezone


class CacheEntry(SQLModel, table=True):
    """Persistent cache storage with expiry support"""

    __tablename__ = "cache_entries"
    __table_args__ = (
        Index("idx_cache_entries_expiry", "expires_at"),
        {"schema": "app_schema"},
    )

    key: str = Field(sa_column=Column(Text, primary_key=True))
    value: t.Dict[str, t.Any] = Field(sa_column=Column(JSONB))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    expires_at: t.Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
