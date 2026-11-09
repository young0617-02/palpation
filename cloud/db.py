from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from cloud.settings import cloud_settings


class Base(DeclarativeBase):
    pass


class EdgeDevice(Base):
    __tablename__ = "edge_devices"
    gw_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    site: Mapped[str | None] = mapped_column(String(128))
    hardware: Mapped[dict] = mapped_column(JSON, default=dict)
    cert_fingerprint: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EdgeHeartbeat(Base):
    __tablename__ = "edge_heartbeat"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gw_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON)


class EdgeEventRow(Base):
    __tablename__ = "edge_events"
    __table_args__ = (UniqueConstraint("event_id", name="uq_event_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    gw_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="received")


class Watermark(Base):
    __tablename__ = "edge_watermarks"
    gw_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_acked_seq: Mapped[int] = mapped_column(Integer, default=0)
    gaps: Mapped[list] = mapped_column(JSON, default=list)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(Text)


class RolloutPlan(Base):
    __tablename__ = "rollout_plan"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    strategy: Mapped[str] = mapped_column(String(32), default="ring")
    rings: Mapped[list] = mapped_column(JSON, default=list)
    frozen: Mapped[bool] = mapped_column(Boolean, default=False)


class RolloutRecord(Base):
    __tablename__ = "rollout_record"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gw_id: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(32))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


class ShadowRow(Base):
    __tablename__ = "edge_shadow"
    gw_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    desired: Mapped[dict] = mapped_column(JSON, default=dict)
    reported: Mapped[dict] = mapped_column(JSON, default=dict)


engine = create_engine(cloud_settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
