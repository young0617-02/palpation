"""MQTT topic family from scheme §2.3 / §6.7."""

from __future__ import annotations


def telemetry(gw_id: str) -> str:
    return f"edge/{gw_id}/telemetry"


def event(gw_id: str) -> str:
    return f"edge/{gw_id}/event"


def event_ack(gw_id: str) -> str:
    return f"edge/{gw_id}/ack"


def diagnosis_preliminary(gw_id: str) -> str:
    return f"edge/{gw_id}/diagnosis/preliminary"


def diagnosis_cloud(gw_id: str) -> str:
    return f"edge/{gw_id}/diagnosis/cloud"


def health(gw_id: str) -> str:
    return f"edge/{gw_id}/health"


def status(gw_id: str) -> str:
    return f"edge/{gw_id}/status"


def shadow_desired(gw_id: str) -> str:
    return f"edge/{gw_id}/shadow/desired"


def shadow_reported(gw_id: str) -> str:
    return f"edge/{gw_id}/shadow/reported"


def manifest(gw_id: str) -> str:
    return f"edge/{gw_id}/manifest"


def command(gw_id: str) -> str:
    return f"edge/{gw_id}/command"


def sparkplug_ddata(group: str, edge_node: str, device_id: str) -> str:
    return f"spBv1.0/{group}/DDATA/{edge_node}/{device_id}"
