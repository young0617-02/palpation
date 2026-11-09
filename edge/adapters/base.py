from __future__ import annotations

import asyncio
from typing import Protocol

from shared.schemas import InternalSample


class Adapter(Protocol):
    name: str

    async def run(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None: ...
