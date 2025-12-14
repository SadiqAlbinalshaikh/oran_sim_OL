__version__ = "0.3.0"

from .core import ORANSimulator
from .config import load_config
from .entities import GroundStation, load_ais_data_polars, Packet, ShipTrafficState
from .communication import (
    CommunicationLayer,
    FastHAPNode,
    FastServerState,
    AggregateQueueState,
    IPPState,
)
from .utils import (
    setup_logging,
    get_timestamp,
    compute_coverage_radius,
    pl_haversine_dist,
    pl_bearing,
    pl_destination_point,
    create_geodesic_circle_poly,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
    HAP_DTYPE,
    SHIP_DTYPE,
    haps_df_to_array,
    ships_df_to_array,
    haversine_distance_km_vectorized,
    compute_slant_range_km_vectorized,
)
from .policies import (
    MobilityPolicy,
    AssociationPolicy,
    SchedulerPolicy,
    AdmissionPolicy,
    AdmissionResult,
    MinimalMobilityPolicy,
    MinimalAssociationPolicy,
    MinimalSchedulerPolicy,
    MinimalAdmissionPolicy,
    create_policies,
)

__all__ = [
    "__version__",
    "ORANSimulator",
    "load_config",
    "GroundStation",
    "load_ais_data_polars",
    "Packet",
    "ShipTrafficState",
    "CommunicationLayer",
    "FastHAPNode",
    "FastServerState",
    "AggregateQueueState",
    "IPPState",
    "setup_logging",
    "get_timestamp",
    "compute_coverage_radius",
    "pl_haversine_dist",
    "pl_bearing",
    "pl_destination_point",
    "create_geodesic_circle_poly",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "HAP_DTYPE",
    "SHIP_DTYPE",
    "haps_df_to_array",
    "ships_df_to_array",
    "haversine_distance_km_vectorized",
    "compute_slant_range_km_vectorized",
    "MobilityPolicy",
    "AssociationPolicy",
    "SchedulerPolicy",
    "AdmissionPolicy",
    "AdmissionResult",
    "MinimalMobilityPolicy",
    "MinimalAssociationPolicy",
    "MinimalSchedulerPolicy",
    "MinimalAdmissionPolicy",
    "create_policies",
    "render_video",
    "render_stats_png",
]


def __getattr__(name):
    if name in ('render_video', 'render_stats_png'):
        from . import video_renderer
        return getattr(video_renderer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
