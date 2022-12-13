import logging
from functools import wraps

from kedro_sagemaker.config import (
    SageMakerMetricsTrackingConfig,
)
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_METRICS,
)

logger = logging.getLogger(__name__)


def sagemaker_metrics(metrics: dict):
    def _decorator(func):
        config = SageMakerMetricsTrackingConfig(metrics)
        setattr(
            func,
            KEDRO_SAGEMAKER_METRICS,
            config,
        )

        @wraps(func)
        def wrapper(*args, **kws):
            # for later use, maybe we will actually need to plug-in custom actions
            return func(*args, **kws)

        return wrapper

    return _decorator
