"""hospital.api — the FastAPI operator API (M2).

A transport shell over the existing engine (doc 07): run lifecycle, live state
streaming, playback control decoupled from wall-clock, operator overrides
through the one ``core.validation.validate()``, live KPIs via the one
``analysis`` fold, and CRN-paired comparison. The public surface is the app
factory; everything else is import-by-module.
"""

from __future__ import annotations

from hospital.api.main import create_app

__all__ = ["create_app"]
