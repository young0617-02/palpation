from __future__ import annotations

import argparse
import asyncio
import json
import logging

log = logging.getLogger("sim-coap")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        from aiocoap import Context, Message, resource
        from aiocoap.numbers.codes import CONTENT
    except ImportError:
        log.warning("aiocoap not installed; CoAP simulator idle")
        await asyncio.Event().wait()
        return

    class Sensor(resource.Resource):
        async def render_get(self, request):
            payload = json.dumps({"temperature": 26.5}).encode()
            return Message(code=CONTENT, payload=payload)

    root = resource.Site()
    root.add_resource(["sensor"], Sensor())
    await Context.create_server_context(root, bind=("0.0.0.0", 5683))
    log.info("coap sensor on :5683/sensor interval hint=%s", args.interval)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
