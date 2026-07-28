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

import math
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


def _json_safe(value: object) -> object:
    """Replace non-finite floats with their repr, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {str(key): _json_safe(item) for key, item in items}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in cast("Sequence[object]", value)]
    return value


async def _render_validation_error(request: Request, exc: Exception) -> Response:
    """A 422 body must be valid JSON even when the body it rejects was not.

    ``json.loads`` accepts the non-standard ``Infinity``/``NaN`` literals, so a
    value pydantic rejects for being non-finite (``ControlCommand.multiplier``)
    reaches the error detail's ``input`` field — where starlette's
    ``json.dumps(allow_nan=False)`` raises and turns a clean rejection into a
    500. Same shape as FastAPI's own handler, with the offending value reported
    as its repr.
    """
    del request
    errors: Sequence[Any] = exc.errors() if isinstance(exc, RequestValidationError) else ()
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(list(errors)))}
    )


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
    app.add_exception_handler(RequestValidationError, _render_validation_error)
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
