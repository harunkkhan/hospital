"""hospital.forecast — statistical and ML forecasting (M3) that feeds the solver's inputs.

The "predict" half of predict-then-optimize. Fits models on sim-generated (later
real) data and publishes their output as **core-typed values** a composition root
threads into the solver's existing input slots.

Depends on ``core`` and ``data`` only. It never imports ``solver``, ``sim``, or
``analysis``, so predictions cannot be pushed downward — they are handed over as
data (:class:`PredictionBundle`) and injected through a ``core``-owned Protocol
(``core.seam.RiskMonitor``). That is what keeps the loop closed without a cycle.

Re-exported here: the model wrappers, the fit entry points, and the adapter
types. Everything else is import-by-module (doc 06 §2).
"""

from hospital.forecast.arrivals import (
    ArrivalIntensityModel,
    SurgeForecast,
    SurgeForecaster,
    fit_arrival_intensity,
    fit_surge_forecaster,
    intensity_from_rates,
)
from hospital.forecast.deterioration import (
    DeteriorationModel,
    RollingDeteriorationMonitor,
    ThresholdChoice,
    fit_deterioration_model,
    news2_for_features,
)
from hospital.forecast.features import (
    ComplaintEncoder,
    FeatureFrame,
    PatientFeatures,
    VitalsWindowFeatures,
    WindowFeatures,
    online_vitals_features,
    patient_features,
    vitals_window_features,
    window_features,
)
from hospital.forecast.model_store import (
    ArtifactMeta,
    ModelStore,
    PredictionBundle,
    bundle_from_models,
    static_bundle,
)
from hospital.forecast.service_time import (
    LognormalParams,
    ServiceTimeKey,
    ServiceTimeRegressor,
    ServiceTimeTable,
    fit_service_time_regressor,
    fit_service_time_table,
    static_service_table,
)
from hospital.forecast.training import (
    GbtParams,
    TrainConfig,
    TrainedBundle,
    ValidationReport,
    WeekData,
    retrain_loop,
    rolling_origin_splits,
    train_all,
    validate_bundle,
)

__all__ = [
    "ArrivalIntensityModel",
    "ArtifactMeta",
    "ComplaintEncoder",
    "DeteriorationModel",
    "FeatureFrame",
    "GbtParams",
    "LognormalParams",
    "ModelStore",
    "PatientFeatures",
    "PredictionBundle",
    "RollingDeteriorationMonitor",
    "ServiceTimeKey",
    "ServiceTimeRegressor",
    "ServiceTimeTable",
    "SurgeForecast",
    "SurgeForecaster",
    "ThresholdChoice",
    "TrainConfig",
    "TrainedBundle",
    "ValidationReport",
    "VitalsWindowFeatures",
    "WeekData",
    "WindowFeatures",
    "bundle_from_models",
    "fit_arrival_intensity",
    "fit_deterioration_model",
    "fit_service_time_regressor",
    "fit_service_time_table",
    "fit_surge_forecaster",
    "intensity_from_rates",
    "news2_for_features",
    "online_vitals_features",
    "patient_features",
    "retrain_loop",
    "rolling_origin_splits",
    "static_bundle",
    "static_service_table",
    "train_all",
    "validate_bundle",
    "vitals_window_features",
    "window_features",
]
