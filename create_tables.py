"""
Creates all database tables from models.py.

Usage:
    python create_tables.py
"""
from sqlalchemy import create_engine
from config import Config
from models import Base


def main():
    engine = create_engine(Config.DATABASE_URL)
    Base.metadata.create_all(engine)
    print(f"Tables created at {Config.DATABASE_URL}")
    print("Tables:", ", ".join(Base.metadata.tables.keys()))


if __name__ == "__main__":
    main()
