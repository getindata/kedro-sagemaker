from unittest.mock import MagicMock

from kedro_sagemaker.config import SageMakerMetricsTrackingConfig
from kedro_sagemaker.constants import KEDRO_SAGEMAKER_METRICS
from kedro_sagemaker.decorators import sagemaker_metrics


def test_can_annotate_sagemaker_metrics():
    fn = MagicMock()
    decorated = sagemaker_metrics(m := {"metric": "regex"})(fn)
    assert (
        hasattr(decorated, KEDRO_SAGEMAKER_METRICS)
        and isinstance(
            (attr := getattr(decorated, KEDRO_SAGEMAKER_METRICS)),
            SageMakerMetricsTrackingConfig,
        )
        and attr.metrics == m
    ), "Invalid attribute assignment"
    decorated()
    fn.assert_called_once()
