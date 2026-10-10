from logging.config import fileConfig
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from sqlmodel import SQLModel
from dsp_decikit.settings import settings
from dsp_decikit.db.models import *  # Import all models so SQLModel can see them

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from our settings
print(f"Using database: {settings.db.database_url}")
print(f"Using schema: {settings.db.pg_schema}")
# config.set_main_option("sqlalchemy.url", settings.db.sync_database_url)
# Don't use set_main_option to avoid ConfigParser interpolation issues with % characters
# config.set_main_option("sqlalchemy.url", settings.db.sync_database_url)

# Set target metadata from SQLModel with schema
target_metadata = SQLModel.metadata

# Configure schema for PostgreSQL - this is crucial for table operations
# Use schema translation map to handle app_schema -> configured schema mapping
# Configuring schema translation: app_schema -> {settings.db.pg_schema}
execution_options = settings.db.get_execution_options()
schema_translate_map = execution_options.get("schema_translate_map", {})


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Use the properly configured database URL with authentication
    url = settings.db.database_url
    print(f"Offline migration URL: {url}")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        version_table_schema=settings.db.pg_schema,
        # Apply schema translation for offline mode
        schema_translate_map=schema_translate_map,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    # Schema translation is handled by execution_options on the engine
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        version_table_schema=settings.db.pg_schema,
        # Apply schema translation for online mode too (double insurance)
        schema_translate_map=schema_translate_map,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """In this scenario we reuse the same engine as the main application
    to ensure identical configuration including SSL, auth hooks, and connection settings.

    """
    # Reuse the main application's engine - already has all the right settings
    from dsp_decikit.db.session import engine as db_engine

    connectable = db_engine

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
