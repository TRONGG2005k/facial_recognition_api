import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
logger.debug("Loaded environment variables from .env")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# Create tables
Base.metadata.create_all(bind=engine)
logger.info("Database tables created or verified")


# ✅ FastAPI dependency
def get_db():
    logger.debug("Creating database session")
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()
