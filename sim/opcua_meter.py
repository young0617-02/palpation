from __future__ import annotations

import asyncio
import logging
import math
import time

from asyncua import Server

log = logging.getLogger("sim-opcua")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")
    idx = await server.register_namespace("http://edge.sim/meter")
    objects = server.nodes.objects
    meter = await objects.add_object(idx, "Meter")
    temp = await meter.add_variable(idx, "Temperature", 45.0)
    curr = await meter.add_variable(idx, "Current", 10.0)
    await temp.set_writable()
    await curr.set_writable()
    t0 = time.time()
    async with server:
        log.info("opcua sim at opc.tcp://0.0.0.0:4840/freeopcua/server/")
        while True:
            t = time.time() - t0
            await temp.write_value(45.0 + math.sin(t / 8.0))
            await curr.write_value(10.0 + 0.2 * math.sin(t))
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
