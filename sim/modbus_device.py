from __future__ import annotations

import argparse
import logging
import math
import time
from threading import Event, Thread

from pymodbus.datastore import (  # type: ignore[attr-defined]
    ModbusDeviceBlock,
    ModbusDeviceContext,
    ModbusServerContext,
)
from pymodbus.server import StartTcpServer

log = logging.getLogger("sim-modbus")


class Holding(ModbusDeviceBlock):
    def __init__(self):
        super().__init__([0] * 16)
        self.t0 = time.time()
        self.fault = False

    def set_fault(self, on: bool) -> None:
        self.fault = on

    def tick(self) -> None:
        t = time.time() - self.t0
        vib = 1.2 + 0.15 * math.sin(t * 2.0)
        if self.fault:
            vib = 6.2 + 0.4 * math.sin(t * 8.0)
        temp = 46.0 + 0.5 * math.sin(t / 10.0) + (12.0 if self.fault else 0.0)
        current = 10.0 + 0.3 * math.sin(t) + (6.0 if self.fault else 0.0)
        self.values[0] = int(vib * 100)
        self.values[1] = int(temp * 10)
        self.values[2] = int(current * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--fault-after", type=float, default=20.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    holding = Holding()
    device = ModbusDeviceContext(hr=holding)
    context = ModbusServerContext(devices={1: device}, single=False)
    stop = Event()

    def loop():
        while not stop.is_set():
            if args.fault_after >= 0 and (time.time() - holding.t0) >= args.fault_after:
                holding.set_fault(True)
            holding.tick()
            time.sleep(1.0 / args.rate)

    Thread(target=loop, daemon=True).start()
    log.info("modbus sim on %s:%s", args.host, args.port)
    StartTcpServer(context=context, address=(args.host, args.port))


if __name__ == "__main__":
    main()
