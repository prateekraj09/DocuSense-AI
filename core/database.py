import logging
import os
import uuid
from typing import AsyncGenerator
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from core.config import settings
from db.base import Base

logger = logging.getLogger(__name__)

# Primary database URL
_current_db_url = settings.get_database_url()
_connect_args = {}
if _current_db_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

async_engine = create_async_engine(
    _current_db_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides an asynchronous database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Helper to initialize database tables and extensions with automatic local fallback."""
    global async_engine, AsyncSessionLocal

    is_postgres = "postgresql" in async_engine.url.drivername

    if is_postgres:
        try:
            # Test PostgreSQL connection with a short timeout
            async with async_engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
                await conn.run_sync(Base.metadata.create_all)
            print("Successfully connected to PostgreSQL (pgvector active).")
        except Exception as e:
            print("\n" + "=" * 70)
            print("⚠️  PostgreSQL server is not running on localhost:5432.")
            print("🔄  Automatically falling back to local SQLite database (docusense_local.db).")
            print("💡  Note: To use PostgreSQL with pgvector, run 'docker compose up' in Docker.")
            print("=" * 70 + "\n")

            # Dispose previous engine
            await async_engine.dispose()

            # Switch to local SQLite database
            fallback_url = "sqlite+aiosqlite:///./docusense_local.db"
            async_engine = create_async_engine(
                fallback_url,
                echo=False,
                future=True,
                connect_args={"check_same_thread": False},
            )
            AsyncSessionLocal.configure(bind=async_engine)

            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("Local database initialized at ./docusense_local.db")
    else:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Seed the demo user (engineer@docusense.ai / SecurePass123!) for instant UI access
    await _seed_demo_user()


async def _seed_demo_user() -> None:
    """Ensures the demo user exists so the UI Sign In works instantly on first launch."""
    from core.security import get_password_hash
    from db.models import User

    demo_email = "engineer@docusense.ai"
    demo_password = "SecurePass123!"

    async with AsyncSessionLocal() as session:
        try:
            res = await session.execute(select(User).where(User.email == demo_email))
            existing = res.scalar_one_or_none()
            if not existing:
                demo_user = User(
                    id=uuid.uuid4(),
                    email=demo_email,
                    hashed_password=get_password_hash(demo_password),
                )
                session.add(demo_user)
                await session.commit()
                print(f"Default demo user registered: {demo_email}")
        except Exception as e:
            logger.warning(f"Demo user seeding skipped: {e}")
