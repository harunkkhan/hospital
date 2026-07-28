"""The FastAPI app factory: routers, CORS, the lifespan-scoped registry, OpenAPI.

Composition only (doc 07 nuances 7.1): no folding, no frame building, no
validation happens here. The :class:`SessionRegistry` and the scenario store
live in the **lifespan**, not a module global, so driver tasks and their SimPy
envs are torn down deterministically when the app (or a ``TestClient`` context)
exits, and test apps never share sessions.

The served ``/openapi.json`` and the codegen dump are ONE document: the wire
manifest (``StreamFrame`` and friends — never an HTTP body, hence otherwise
missing) is merged into this app's own OpenAPI schema, and ``codegen`` dumps
exactly ``app.openapi()``.

CORS is permissive for localhost dev only (the Vite proxy fronts it); auth is
out of scope in M2.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hospital.api import overrides, runs, sessions, stream
from hospital.api.runs import ScenarioStore
from hospital.api.sessions import SessionRegistry
from hospital.api.wire import merge_wire_schemas

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

API_TITLE = "hospital-api"
API_VERSION = "0.1.0"

_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class HospitalAPI(FastAPI):
    """FastAPI whose OpenAPI document carries the merged wire manifest.

    Overriding :meth:`openapi` (rather than post-processing a dump) keeps the
    live ``/openapi.json`` and the ``codegen`` output byte-identical — one
    schema, two exits.
    """

    def openapi(self) -> dict[str, Any]:
        if not self.openapi_schema:
            self.openapi_schema = merge_wire_schemas(super().openapi())
        return self.openapi_schema


def create_app(*, scenario_dir: str | Path | None = "scenarios") -> FastAPI:
    """Build the operator API app.

    ``scenario_dir`` seeds the scenario store from ``*.yaml`` files at startup
    (``None`` starts empty — used by codegen and tests that register their own).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.registry = SessionRegistry()
        app.state.scenarios = (
            ScenarioStore.from_dir(Path(scenario_dir))
            if scenario_dir is not None
            else ScenarioStore()
        )
        try:
            yield
        finally:
            await app.state.registry.shutdown()

    app = HospitalAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_DEV_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(runs.router)
    app.include_router(sessions.router)
    app.include_router(overrides.router)
    app.include_router(stream.router)
    return app


__all__ = ["API_TITLE", "API_VERSION", "create_app"]
