from __future__ import annotations

import argparse
import json
import math
import time

import paho.mqtt.client as mqtt

from shared.topics import sparkplug_ddata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--gw", default="gw-sim-001")
    parser.add_argument("--device", default="pump-de-01")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--fault-after", type=float, default=25.0)
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sim-pump")
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    topic = sparkplug_ddata("plant1", args.gw, args.device)
    t0 = time.time()
    seq = 0
    while True:
        t = time.time() - t0
        fault = args.fault_after >= 0 and t >= args.fault_after
        vib = 6.3 if fault else 1.15 + 0.1 * math.sin(t)
        payload = {
            "timestamp": int(time.time() * 1000),
            "seq": seq,
            "device_id": args.device,
            "metrics": [
                {"name": "vibration_rms", "type": "Float", "value": vib},
                {"name": "temperature", "type": "Float", "value": 47.0 if not fault else 78.0},
                {"name": "current", "type": "Float", "value": 10.2 if not fault else 16.5},
            ],
        }
        client.publish(topic, json.dumps(payload), qos=1)
        seq += 1
        time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
