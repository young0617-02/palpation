"""cloud/db.py 单元测试：SQLAlchemy 模型与 init_db，使用 sqlite 内存库。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import (
    Base,
    EdgeDevice,
    EdgeEventRow,
    ModelRegistry,
    RolloutPlan,
    RolloutRecord,
    ShadowRow,
    Watermark,
    init_db,
)


@pytest.fixture()
def session():
    """每个用例独立的内存 sqlite session。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        yield s


# --------------------------------------------------------------------------- #
# init_db
# --------------------------------------------------------------------------- #

def test_init_db_creates_all_tables(monkeypatch):
    """init_db 在内存引擎上建表，metadata 包含全部表。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    monkeypatch.setattr("cloud.db.engine", engine)
    init_db()
    expected = {
        "edge_devices",
        "edge_heartbeat",
        "edge_events",
        "edge_watermarks",
        "model_registry",
        "rollout_plan",
        "rollout_record",
        "edge_shadow",
    }
    assert expected.issubset(Base.metadata.tables.keys())


def test_base_is_declarative():
    """Base 为 DeclarativeBase 子类，metadata 可建表。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    assert "edge_devices" in Base.metadata.tables


# --------------------------------------------------------------------------- #
# EdgeDevice
# --------------------------------------------------------------------------- #

def test_insert_and_query_device(session):
    dev = EdgeDevice(gw_id="gw-001", site="shanghai", hardware={"cpu": "arm64"},
                      cert_fingerprint="fp-abc")
    session.add(dev)
    session.commit()
    got = session.query(EdgeDevice).filter_by(gw_id="gw-001").one()
    assert got.site == "shanghai"
    assert got.hardware == {"cpu": "arm64"}
    assert got.cert_fingerprint == "fp-abc"


def test_device_primary_key_unique(session):
    session.add(EdgeDevice(gw_id="gw-dup"))
    session.commit()
    session.add(EdgeDevice(gw_id="gw-dup"))
    with pytest.raises(Exception):
        session.commit()


def test_device_optional_fields_default_none(session):
    dev = EdgeDevice(gw_id="gw-002")
    session.add(dev)
    session.commit()
    got = session.query(EdgeDevice).filter_by(gw_id="gw-002").one()
    assert got.site is None
    assert got.cert_fingerprint is None
    assert got.hardware == {}


# --------------------------------------------------------------------------- #
# EdgeEventRow
# --------------------------------------------------------------------------- #

def test_insert_and_query_event(session):
    ev = EdgeEventRow(event_id="evt-1", gw_id="gw-001", seq=1, payload={"v": 6.1})
    session.add(ev)
    session.commit()
    got = session.query(EdgeEventRow).filter_by(event_id="evt-1").one()
    assert got.seq == 1
    assert got.status == "received"  # default
    assert got.payload == {"v": 6.1}


def test_event_id_unique_constraint(session):
    session.add(EdgeEventRow(event_id="dup", gw_id="gw-001", seq=1, payload={}))
    session.commit()
    session.add(EdgeEventRow(event_id="dup", gw_id="gw-001", seq=2, payload={}))
    with pytest.raises(Exception):
        session.commit()


# --------------------------------------------------------------------------- #
# Watermark
# --------------------------------------------------------------------------- #

def test_watermark_insert_update_and_gaps(session):
    wm = Watermark(gw_id="gw-001", last_acked_seq=5, gaps=[6, 7])
    session.add(wm)
    session.commit()
    got = session.query(Watermark).filter_by(gw_id="gw-001").one()
    assert got.last_acked_seq == 5
    assert got.gaps == [6, 7]
    # 更新
    got.last_acked_seq = 10
    got.gaps = []
    session.commit()
    got2 = session.query(Watermark).filter_by(gw_id="gw-001").one()
    assert got2.last_acked_seq == 10
    assert got2.gaps == []


def test_watermark_default_values(session):
    wm = Watermark(gw_id="gw-003")
    session.add(wm)
    session.commit()
    got = session.query(Watermark).filter_by(gw_id="gw-003").one()
    assert got.last_acked_seq == 0
    assert got.gaps == []


# --------------------------------------------------------------------------- #
# ModelRegistry / ShadowRow / RolloutPlan / RolloutRecord
# --------------------------------------------------------------------------- #

def test_model_registry_and_shadow(session):
    mr = ModelRegistry(model_id="anomaly", version="0.1.0", sha256="abc123",
                       manifest={"k": 1}, artifact_path="/tmp/a.onnx")
    session.add(mr)
    sh = ShadowRow(gw_id="gw-001", desired={"v": "0.2"}, reported={"v": "0.1"})
    session.add(sh)
    session.commit()
    assert session.query(ModelRegistry).count() == 1
    got_sh = session.query(ShadowRow).one()
    assert got_sh.desired == {"v": "0.2"}
    assert got_sh.reported == {"v": "0.1"}


def test_rollout_plan_and_record(session):
    plan = RolloutPlan(model_id="anomaly", version="0.2.0", strategy="ring",
                       rings=["gw-001", "gw-002"])
    session.add(plan)
    session.commit()
    rec = RolloutRecord(gw_id="gw-001", model_id="anomaly", version="0.2.0",
                        state="succeeded", detail={"probe": "ok"})
    session.add(rec)
    session.commit()
    got_plan = session.query(RolloutPlan).one()
    assert got_plan.frozen is False
    assert got_plan.rings == ["gw-001", "gw-002"]
    got_rec = session.query(RolloutRecord).one()
    assert got_rec.state == "succeeded"
    assert got_rec.detail == {"probe": "ok"}