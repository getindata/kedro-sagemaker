import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import pytest
from kedro.pipeline import pipeline, Pipeline, node

from kedro_sagemaker.constants import KEDRO_SAGEMAKER_RUNNER_CONFIG
from kedro_sagemaker.datasets import CloudpickleDataset, DistributedCloudpickleDataset
from kedro_sagemaker.runner import KedroSageMakerRunnerConfig, SageMakerPipelinesRunner


def identity(x):
    return x


@pytest.fixture(params=[CloudpickleDataset, DistributedCloudpickleDataset])
def patched_cloudpickle_dataset(request):
    with TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / (uuid4().hex + ".bin")
    with patch.object(
        CloudpickleDataset,
        "_get_target_path",
        return_value=str(target_path.absolute()),
    ):
        yield request.param("bucket", "unit_tests_ds", uuid4().hex)


@pytest.fixture(params=(True, False))
def patched_sagemaker_runner(patched_cloudpickle_dataset, request):
    is_distributed = request.param
    distributed_envs = {"RANK": "0"} if is_distributed else {}
    cfg = KedroSageMakerRunnerConfig(
        bucket=f"unit-tests-bucket-{uuid4().hex}", run_id=uuid4().hex
    )
    with patch.dict(
        os.environ,
        {KEDRO_SAGEMAKER_RUNNER_CONFIG: cfg.json(), **distributed_envs},
        clear=False,
    ):
        yield SageMakerPipelinesRunner()


@pytest.fixture()
def dummy_pipeline() -> Pipeline:

    return pipeline(
        [
            node(identity, inputs="input_data", outputs="i2", name="node1"),
            node(identity, inputs="i2", outputs="i3", name="node2"),
            node(identity, inputs="i3", outputs="output_data", name="node3"),
        ]
    )
