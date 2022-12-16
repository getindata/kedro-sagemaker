from unittest.mock import patch, MagicMock

import pytest

from kedro_sagemaker.client import SageMakerClient
from kedro_sagemaker.config import _CONFIG_TEMPLATE
from kedro_sagemaker.generator import KedroSageMakerGenerator


@pytest.mark.parametrize("is_local", (True, False), ids=("local", "not local"))
@pytest.mark.parametrize(
    "wait_for_completion",
    (True, False),
    ids=("wait", "no wait"),
)
@pytest.mark.parametrize(
    "callback", (True, False), ids=("with callback", "without callback")
)
def test_client_can_invoke_run(is_local, wait_for_completion, callback):
    mock_pipeline = MagicMock()
    started_pipeline = MagicMock()
    mock_pipeline.start = MagicMock(return_value=started_pipeline)
    on_pipeline_started = MagicMock() if callback else None

    sm_client = SageMakerClient(mock_pipeline, "role")
    result = sm_client.run(
        is_local=is_local,
        wait_for_completion=wait_for_completion,
        on_pipeline_started=on_pipeline_started,
    )

    assert result, "Pipeline should not fail on mock"
    mock_pipeline.start.assert_called_once()

    if callback:
        on_pipeline_started.assert_called_once()

    if wait_for_completion and not is_local:
        started_pipeline.wait.assert_called_once()
