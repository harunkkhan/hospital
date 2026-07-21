"""Domain-named, stateless thin wrappers over ``core.rng`` samplers.

Every helper takes a ``numpy.random.Generator`` obtained upstream from
``RandomStreams.substream(...)`` — none of them creates or seeds one (doc 02
§2.6). There is deliberately **one** categorical idiom: ``sample_esi``,
``sample_complaint``, ``sample_arrival_mode``, and even ``bernoulli`` all route
through :func:`hospital.core.sample_categorical`, so booleans and modes are
never special-cased with a second RNG path.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hospital.core import ArrivalMode, EsiAcuity, WorkupNeeds, sample_categorical
from hospital.data.scenario import WorkupProfile


def sample_esi(g: np.random.Generator, mix: Mapping[EsiAcuity, float]) -> EsiAcuity:
    """Draw an ESI acuity from ``mix``."""
    return sample_categorical(g, mix)


def sample_complaint(g: np.random.Generator, mix: Mapping[str, float]) -> str:
    """Draw a complaint label from ``mix``."""
    return sample_categorical(g, mix)


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
    g: np.random.Generator, profile: WorkupProfile, *, esi_scale: float = 1.0
) -> WorkupNeeds:
    """Compose the workload primitives above into a pre-sampled ``WorkupNeeds``.

    ``provider_visits`` is floored at 1 (a visit always has at least one
    provider contact); ``esi_scale`` multiplies each *mean* before sampling so
    higher acuity monotonically increases expected intensity.
    """
    provider_visits = 1 + sample_count(g, (profile.provider_visits_mean - 1.0) * esi_scale)
    nurse_visits = sample_count(g, profile.nurse_visits_mean * esi_scale)
    # Iterated in the profile's own (insertion) order — deterministic, since
    # `imaging_prob` comes from a loaded YAML mapping, never a re-sorted copy.
    imaging = tuple(zt for zt, p in profile.imaging_prob.items() if bernoulli(g, p))
    labs = sample_count(g, profile.labs_mean)
    procedures = int(bernoulli(g, profile.procedure_prob))
    return WorkupNeeds(
        provider_visits=provider_visits,
        nurse_visits=nurse_visits,
        imaging=imaging,
        labs=labs,
        procedures=procedures,
    )


__all__ = [
    "bernoulli",
    "sample_arrival_mode",
    "sample_complaint",
    "sample_count",
    "sample_esi",
    "sample_workup",
]
