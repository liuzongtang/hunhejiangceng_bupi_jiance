"""
Database engine and session management.

Uses SQLAlchemy 2.0 async with PostgreSQL (production) or SQLite (dev/test).
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_config


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


# Global engine and session factory
_engine = None
_session_factory = None


async def get_engine():
    """Get or create the async SQLAlchemy engine."""
    global _engine, _session_factory

    if _engine is None:
        config = get_config()
        # Use SQLite for development, PostgreSQL for production
        if config.environment == "development":
            db_url = config.database.sqlite_url
        else:
            db_url = config.database.url

        _engine = create_async_engine(
            db_url,
            echo=config.database.echo,
            future=True,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Create all tables
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async database session."""
    global _session_factory
    if _session_factory is None:
        await get_engine()

    async with _session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database and create all tables."""
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
