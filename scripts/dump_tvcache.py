"""Dump every Tool Call Graph from a running TVCache server to JSON.

    python scripts/dump_tvcache.py out.json [--host 127.0.0.1 --port 7878]
"""
import argparse
import asyncio
import json

from tvclient import TVCacheClient


async def main(a):
    async with TVCacheClient(a.host, a.port) as c:
        dump = await c.dump_all()
    with open(a.out, "w") as fh:
        json.dump(dump, fh)
    print(f"dumped {len(dump) if hasattr(dump, '__len__') else '?'} task graphs -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7878)
    asyncio.run(main(ap.parse_args()))
