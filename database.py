from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI 의존성: 요청마다 DB 세션을 열고 닫는다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """테이블 생성 (초기 개발용. 운영 단계에서는 Alembic 마이그레이션 권장)."""
    import models  # noqa: F401  모델을 Base.metadata에 등록

    Base.metadata.create_all(bind=engine)
    _sync_schema()


def _sync_schema():
    """create_all은 기존 테이블을 변경하지 않으므로, 모델에 새로 추가된 컬럼과
    PostgreSQL enum 값을 보충한다. (간이 마이그레이션 — 스키마가 복잡해지면 Alembic으로 전환)"""
    import enum

    from sqlalchemy import Enum, inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                default = col.default.arg if col.default is not None and col.default.is_scalar else None
                if isinstance(default, enum.Enum):
                    default = default.value
                default_sql = ""
                if isinstance(default, str):
                    default_sql = " DEFAULT '" + default.replace("'", "''") + "'"
                elif isinstance(default, (int, float)):
                    default_sql = f" DEFAULT {default}"
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col_type}{default_sql}'))

        if engine.dialect.name == "postgresql":
            for table in Base.metadata.sorted_tables:
                for col in table.columns:
                    if isinstance(col.type, Enum) and col.type.native_enum and col.type.name:
                        for value in col.type.enums:
                            conn.execute(text(
                                f"ALTER TYPE {col.type.name} ADD VALUE IF NOT EXISTS '{value}'"
                            ))
