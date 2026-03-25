"""
Database configuration và connection management.
"""

import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.debug("Loaded environment variables from .env")

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SSL_CA_PATH = "../../certs/ca.pem"
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/face?ssl_ca={SSL_CA_PATH}&ssl_verify_cert=true&ssl_verify_identity=true"

# SSL certificate path (container path)


# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=True,
    
    pool_pre_ping=True,
    pool_recycle=1800,
    
      # Pool size
    pool_size=5,
    max_overflow=10,
    
    connect_args={
        "ssl": {
            "ssl_ca": SSL_CA_PATH
        }
    } if os.path.exists(SSL_CA_PATH) else {}
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

def init_db() -> None:
    """Khởi tạo database tables. Gọi sau khi app start."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created or verified")


def get_db() -> Session:
    """
    Dependency cung cấp database session.
    
    Yields:
        Session: Database session
        
    Example:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    logger.debug("Creating database session")
    db = SessionLocal()
    try:
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()
