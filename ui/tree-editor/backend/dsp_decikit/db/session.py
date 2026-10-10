import typing as t
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel

from dsp_decikit.settings import settings

engine = create_async_engine(
    settings.db.database_url,
    echo=settings.db.echo,
    echo_pool=settings.db.echo_pool,
    pool_size=settings.db.max_pool_size,
    max_overflow=settings.db.max_overflow,
    pool_timeout=settings.db.pool_timeout,
    pool_recycle=settings.db.pool_recycle,
    pool_pre_ping=settings.db.pool_pre_ping,
    pool_reset_on_return=settings.db.pool_reset_on_return,
    connect_args=settings.db.get_connect_args(),
    execution_options=settings.db.get_execution_options(),
    future=True,
)


async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@event.listens_for(engine.sync_engine, "do_connect")
def auth_event_hook(dialect, conn_rec, cargs, cparams):
    settings.db.auth.event_do_connect_hook(cparams, settings.db)


# I don think this is needed but will keep it here for the future
# @event.listens_for(engine.sync_engine, "handle_error")
# def retry_on_auth_error(exception_context):
#     print(exception_context.original_exception)
#     """Handle database errors and retry on auth failures"""
#     exception = exception_context.original_exception

#     if isinstance(exception, OperationalError) and is_auth_error(exception):
#         logger.warning(f"Authentication error detected: {exception}")
#         logger.info("Invalidating AWS IAM token and will retry on next connection")

#         # Invalidate the token to force refresh
#         aws_auth.invalidate_token()

#     # Mark connection as should_invalidate_pool to force pool refresh
#     if hasattr(exception_context, 'connection_invalidated'):
#         exception_context.connection_invalidated = True
#         logger.info("Token invalidated, next connection attempt will use fresh token")


@asynccontextmanager
async def db_session() -> t.AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        # Testing if removing the close allows session_maker to be more efficient with pools
        # finally:
        #     await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
