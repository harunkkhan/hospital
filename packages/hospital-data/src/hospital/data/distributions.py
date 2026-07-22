"""Domain-named, stateless thin wrappers over ``core.rng`` samplers.

Every helper takes a ``numpy.random.Generator`` obtained upstream from
``RandomStreams.substream(...)`` — none of them creates or seeds one (doc 02
§2.6). There is deliberately **one** categorical idiom: ``sample_esi``,
``sample_complaint``, ``sample_arrival_mode``, and even ``bernoulli`` all route
through :func:`hospital.core.sample_categorical`, so booleans and modes are
never special-cased with a second RNG path.

**Canonical iteration order.** ``core.rng.sample_categorical`` consumes its
mapping in insertion order ("the caller owns ordering"), so these wrappers
sample over **sorted** keys. That makes the draw a pure function of the mix's
*content*, never of how the mapping happened to be built — a dump+reload of an
equal scenario (whose YAML codec sorts keys) therefore maps the same draw to
the same outcome, which is the CRN-across-serialization invariant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from hospital.core import ArrivalMode, EsiAcuity, WorkupNeeds, sample_categorical
from hospital.data.scenario import WorkupProfile

# ``sample_workup`` draws every component on its own content-addressed
# substream; the caller supplies the keyed-generator factory (built from
# ``RandomStreams.substream``), so this module still never seeds anything.
WorkupSubstream = Callable[[str], np.random.Generator]


def sample_esi(g: np.random.Generator, mix: Mapping[EsiAcuity, float]) -> EsiAcuity:
    """Draw an ESI acuity from ``mix`` (sampled over sorted keys — see module doc)."""
    return sample_categorical(g, dict(sorted(mix.items())))


def sample_complaint(g: np.random.Generator, mix: Mapping[str, float]) -> str:
    """Draw a complaint label from ``mix`` (sampled over sorted keys — see module doc)."""
    return sample_categorical(g, dict(sorted(mix.items())))


def sample_arrival_mode(g: np.random.Generator, ambulance_fraction: float) -> ArrivalMode:
    """Draw ``AMBULANCE`` with probability ``ambulance_fraction``, else ``WALK_IN``."""
    return sample_categorical(
        g,
        {ArrivalMode.AMBULANCE: ambulance_fraction, ArrivalMode.WALK_IN: 1.0 - ambulance_fraction},
    )


def bernoulli(g: np.random.Generator, p: float) -> bool:
    """A single Bernoulli(``p``) draw, routed through the one categorical idiom."""
    return sample_categorical(g, {True: p, False: 1.0 - p})


def sample_count(g: np.random.Generator, mean: float, *, minimum: int = 0) -> int:
    """A Poisson(``mean``) draw, floored at ``minimum`` (never negative)."""
    lam = max(0.0, mean)
    return max(minimum, int(g.poisson(lam)))


def sample_workup(
    substream_for: WorkupSubstream, profile: WorkupProfile, *, esi_scale: float = 1.0
) -> WorkupNeeds:
    """Compose the workload primitives above into a pre-sampled ``WorkupNeeds``.

    ``provider_visits`` is floored at 1 (a visit always has at least one
    provider contact); ``esi_scale`` multiplies each *mean* before sampling so
    higher acuity monotonically increases expected intensity.

    Every component draws on its **own** content-addressed substream
    (``provider`` / ``nurse`` / ``imaging_<zone>`` / ``labs`` / ``procedures``),
    obtained via ``substream_for`` — so editing one component of a profile
    (adding an imaging entry, even a zero-probability one) never shifts any
    other component's draws for the same patient. Imaging entries are keyed
    per zone type and evaluated in sorted order, so their draws are independent
    of both mapping insertion order and of each other.
    """
    provider_visits = 1 + sample_count(
        substream_for("provider"), (profile.provider_visits_mean - 1.0) * esi_scale
    )
    nurse_visits = sample_count(substream_for("nurse"), profile.nurse_visits_mean * esi_scale)
    imaging = tuple(
        zt
        for zt in sorted(profile.imaging_prob)
        if bernoulli(substream_for(f"imaging_{zt.value}"), profile.imaging_prob[zt])
    )
    labs = sample_count(substream_for("labs"), profile.labs_mean * esi_scale)
    procedures = int(bernoulli(substream_for("procedures"), profile.procedure_prob))
    return WorkupNeeds(
        provider_visits=provider_visits,
        nurse_visits=nurse_visits,
        imaging=imaging,
        labs=labs,
        procedures=procedures,
    )


__all__ = [
    "WorkupSubstream",
    "bernoulli",
    "sample_arrival_mode",
    "sample_complaint",
    "sample_count",
    "sample_esi",
    "sample_workup",
]
