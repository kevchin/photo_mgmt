"""
Database configuration for embedding evolution system.
Supports both legacy (source) and evolution (target) databases.
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector

# Database URLs
LEGACY_DATABASE_URL = os.getenv(
    "LEGACY_DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/image_archive"
)
EVOLUTION_DATABASE_URL = os.getenv(
    "EVOLUTION_DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/image_archive_evolution"
)

# Create engines
legacy_engine = create_engine(LEGACY_DATABASE_URL, pool_pre_ping=True)
evolution_engine = create_engine(EVOLUTION_DATABASE_URL, pool_pre_ping=True)

# Session factories
LegacySessionLocal = sessionmaker(bind=legacy_engine, autocommit=False, autoflush=False)
EvolutionSessionLocal = sessionmaker(bind=evolution_engine, autocommit=False, autoflush=False)


def get_legacy_session() -> Session:
    """Get a session for the legacy database."""
    session = LegacySessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_evolution_session() -> Session:
    """Get a session for the evolution database."""
    session = EvolutionSessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_pgvector_installed(session: Session) -> bool:
    """Check if pgvector extension is installed in the database."""
    try:
        result = session.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        ).scalar()
        return result
    except Exception:
        return False


def install_pgvector(session: Session):
    """Install pgvector extension if not already installed."""
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()


def create_evolution_database_if_not_exists():
    """Create the evolution database if it doesn't exist."""
    # Connect to default postgres database to create new database
    default_engine = create_engine(
        "postgresql://postgres:postgres@localhost:5432/postgres",
        isolation_level="AUTOCOMMIT"
    )
    
    with default_engine.connect() as conn:
        # Check if database exists
        result = conn.execute(
            text("""
                SELECT 1 FROM pg_database WHERE datname = 'image_archive_evolution'
            """)
        ).fetchone()
        
        if not result:
            conn.execute(text("CREATE DATABASE image_archive_evolution"))
            print("Created evolution database: image_archive_evolution")
        else:
            print("Evolution database already exists: image_archive_evolution")
