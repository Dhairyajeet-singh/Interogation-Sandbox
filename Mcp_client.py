"""
Mcp_client.py   -  Stage 8
--------------------------
Connects to the forensic server, discovers what it offers, and calls it.

The point of going through a client rather than importing the functions
is that the agent has to DISCOVER the tools rather than have them
hard-coded. Add a tool to Mcp_server.py and the detective can use it with
no change here or in the game loop.

Two transports:

    ForensicsClient()                 spawns Mcp_server.py over stdio
    ForensicsClient(http="http://127.0.0.1:8765")   talks to the HTTP one

Both expose the same three calls: list_tools(), call(name, **args),
read_resource(uri).

The MCP SDK is async, so the stdio path runs its own event loop inside
synchronous methods. That keeps the game loop simple - nothing else in
the project has to become async.
"""

import asyncio
import json
import sys
from pathlib import Path

SERVER = Path(__file__).with_name("Mcp_server.py")


def _unwrap(result):
    """
    MCP returns content blocks. Pull out the useful payload: parsed JSON
    when the tool returned structured data, otherwise the text.
    """
    if hasattr(result, "structuredContent") and result.structuredContent:
        sc = result.structuredContent
        # the SDK wraps a bare return value under "result"
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        return sc

    blocks = getattr(result, "content", None) or []
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    if not texts:
        return None
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except (ValueError, TypeError):
        return joined


class ForensicsClient:
    """Synchronous wrapper over either transport."""

    def __init__(self, http=None, server_path=SERVER, python=None):
        self.http = http.rstrip("/") if http else None
        self.server_path = str(server_path)
        self.python = python or sys.executable

    # ---------------------------------------------------- stdio
    async def _with_session(self, fn):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.python,
            args=[self.server_path],
            cwd=str(Path(self.server_path).parent),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # already inside a loop (e.g. a notebook): use a private one
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ---------------------------------------------------- public API
    def list_tools(self):
        """[(name, description), ...] - what the server says it can do."""
        if self.http:
            import httpx
            data = httpx.get(f"{self.http}/tools", timeout=10).json()
            return [(t["name"], t["doc"]) for t in data["tools"]]

        async def go(session):
            res = await session.list_tools()
            return [(t.name, (t.description or "").strip()) for t in res.tools]

        return self._run(self._with_session(go))

    def call(self, name, **args):
        """Invoke a tool by name."""
        if self.http:
            import httpx
            r = httpx.post(f"{self.http}/tools/{self._plain(name)}",
                           json={"args": args}, timeout=30)
            r.raise_for_status()
            return r.json()["result"]

        async def go(session):
            return _unwrap(await session.call_tool(name, args))

        return self._run(self._with_session(go))

    def read_resource(self, uri):
        """Read one of the server's resources."""
        if self.http:
            import httpx
            r = httpx.get(f"{self.http}/resources", params={"uri": uri},
                          timeout=10)
            r.raise_for_status()
            return r.json()["text"]

        async def go(session):
            res = await session.read_resource(uri)
            return "\n".join(c.text for c in res.contents
                             if hasattr(c, "text"))

        return self._run(self._with_session(go))

    # the MCP tools carry a _tool suffix; the HTTP ones do not
    @staticmethod
    def _plain(name):
        return name[:-5] if name.endswith("_tool") else name


# ======================================================================
# a tiny facade the game loop can use without knowing about MCP at all
# ======================================================================

class Forensics:
    """
    What the detective agent sees. Discovers the tool names on first use,
    so the game never hard-codes them.
    """

    def __init__(self, client=None):
        self.client = client or ForensicsClient()
        self._names = None

    def available(self):
        if self._names is None:
            self._names = {n: d for n, d in self.client.list_tools()}
        return self._names

    def _resolve(self, want):
        for name in self.available():
            if name == want or name == f"{want}_tool":
                return name
        raise KeyError(f"server offers no tool like {want!r}: "
                       f"{sorted(self.available())}")

    def check_alibi(self, suspect_id, place, at):
        return self.client.call(self._resolve("check_alibi"),
                                suspect_id=suspect_id, place=place, at=at)

    def verify_statement(self, suspect_id, statement):
        return self.client.call(self._resolve("verify_statement"),
                                suspect_id=suspect_id, statement=statement)

    def who_was_at(self, place, at=None):
        return self.client.call(self._resolve("who_was_at"),
                                place=place, at=at)

    def movements_of(self, suspect_id):
        return self.client.call(self._resolve("movements_of"),
                                suspect_id=suspect_id)

    def lookup_evidence(self, query):
        return self.client.call(self._resolve("lookup_evidence"), query=query)