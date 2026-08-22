"""
BridgeGuardian AI — Database Setup
SQLAlchemy engine, connection pool, session factory, and database dependency.
Supports PostgreSQL in production with pool management, and SQLite in development.
"""
from __future__ import annotations

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import get_settings

logger = logging.getLogger("bridgeguardian.database")
settings = get_settings()

is_sqlite = "sqlite" in settings.database_url.lower()

engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}

if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60.0}
else:
    engine_kwargs["pool_size"] = settings.db_pool_size
    engine_kwargs["max_overflow"] = settings.db_max_overflow
    engine_kwargs["pool_recycle"] = settings.db_pool_recycle

engine = create_engine(settings.database_url, **engine_kwargs)

if is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency providing an isolated database session per request.
    Automatically rolls back on exception and closes on completion.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as exc:
        db.rollback()
        raise exc
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables and run lightweight column migrations on application startup."""
    from backend.core import models  # noqa: F401
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database tables initialized successfully ({'SQLite' if is_sqlite else 'PostgreSQL'}).")

        # Lightweight migration for newly added columns on prediction_records table
        with engine.connect() as conn:
            from sqlalchemy import inspect, text
            inspector = inspect(engine)
            if "prediction_records" in inspector.get_table_names():
                columns = {col["name"] for col in inspector.get_columns("prediction_records")}
                new_cols = {
                    "analysis_type": "VARCHAR(50)",
                    "campaign_id": "INTEGER",
                    "image_count": "INTEGER",
                    "summary_report": "TEXT",
                    "status": "VARCHAR(50)",
                }
                for col_name, col_type in new_cols.items():
                    if col_name not in columns:
                        try:
                            conn.execute(text(f"ALTER TABLE prediction_records ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                            logger.info(f"Added column {col_name} to prediction_records table.")
                        except Exception as alter_err:
                            logger.warning(f"Could not add column {col_name} to prediction_records: {alter_err}")
    except Exception as e:
        logger.error(f"Database initialization warning: {e}")
