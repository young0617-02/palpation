from __future__ import annotations

import json
import logging

import paho.mqtt.client as mqtt

from cloud.db import EdgeEventRow, EdgeHeartbeat, SessionLocal, ShadowRow, Watermark
from cloud.diagnosis import depth_diagnosis
from shared import topics
from shared.schemas import AckPayload, EdgeEvent, EdgeHealth

log = logging.getLogger(__name__)


def attach(client: mqtt.Client) -> None:
    def on_connect(c, userdata, flags, reason_code, properties=None):
        c.subscribe("edge/+/event", qos=1)
        c.subscribe("edge/+/health", qos=0)
        c.subscribe("edge/+/shadow/reported", qos=1)
        c.subscribe("edge/+/status", qos=1)

    def on_message(c, userdata, msg):
        parts = msg.topic.split("/")
        if len(parts) < 3:
            return
        gw_id = parts[1]
        kind = parts[2]
        try:
            if kind == "event":
                _ingest_event(c, EdgeEvent.model_validate_json(msg.payload))
            elif kind == "health":
                health = EdgeHealth.model_validate_json(msg.payload)
                with SessionLocal() as s:
                    s.add(
                        EdgeHeartbeat(
                            gw_id=health.gw_id, ts=health.ts, payload=health.model_dump(mode="json")
                        )
                    )
                    s.commit()
            elif kind == "shadow" and len(parts) > 3 and parts[3] == "reported":
                reported = json.loads(msg.payload.decode("utf-8"))
                with SessionLocal() as s:
                    row = s.get(ShadowRow, gw_id) or ShadowRow(gw_id=gw_id, desired={}, reported={})
                    row.reported = reported
                    s.merge(row)
                    s.commit()
        except Exception:
            log.exception("cloud mqtt message failed topic=%s", msg.topic)

    client.on_connect = on_connect
    client.on_message = on_message


def _ingest_event(client: mqtt.Client, event: EdgeEvent) -> None:
    with SessionLocal() as s:
        existing = s.query(EdgeEventRow).filter_by(event_id=event.event_id).one_or_none()
        if existing is None:
            s.add(
                EdgeEventRow(
                    event_id=event.event_id,
                    gw_id=event.gw_id,
                    seq=event.seq,
                    payload=event.model_dump(mode="json"),
                )
            )
        wm = s.get(Watermark, event.gw_id) or Watermark(
            gw_id=event.gw_id, last_acked_seq=0, gaps=[]
        )
        expected = wm.last_acked_seq + 1
        if event.seq > expected:
            gaps = list(wm.gaps or [])
            gaps.append({"from": expected, "to": event.seq - 1})
            wm.gaps = gaps
            wm.last_acked_seq = event.seq
        elif event.seq == expected:
            wm.last_acked_seq = event.seq
        s.merge(wm)
        s.commit()
        acked = wm.last_acked_seq
    client.publish(
        topics.event_ack(event.gw_id),
        AckPayload(gw_id=event.gw_id, last_acked_seq=acked).model_dump_json(),
        qos=1,
    )
    report = depth_diagnosis(event)
    client.publish(topics.diagnosis_cloud(event.gw_id), report.model_dump_json(), qos=1)
