import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from kedro.pipeline import Pipeline, node, pipeline

from kedro_sagemaker.cli_functions import get_context_and_pipeline
from kedro_sagemaker.constants import KEDRO_SAGEMAKER_RUNNER_CONFIG
from kedro_sagemaker.datasets import (
    CloudpickleDataset,
    DistributedCloudpickleDataset,
)
from kedro_sagemaker.runner import (
    KedroSageMakerRunnerConfig,
    SageMakerPipelinesRunner,
)
from kedro_sagemaker.utils import CliContext
from tests.utils import identity


@pytest.fixture()
def patched_kedro_package():
    with patch("kedro.framework.project.PACKAGE_NAME", "tests") as patched_package:
        original_dir = os.getcwd()
        os.chdir("tests")
        yield patched_package
        os.chdir(original_dir)


@pytest.fixture()
def cli_context() -> CliContext:
    metadata = MagicMock()
    metadata.package_name = "tests"
    return CliContext("base", metadata)


@pytest.fixture(autouse=True)
def patch_boto3_session():
    with patch("boto3.Session"):
        yield


@pytest.fixture()
def context_manager_and_pipeline(patched_kedro_package, dummy_pipeline):
    with patch(
        "kedro.framework.project.pipelines",
        return_value={"__default__": dummy_pipeline},
    ), patch("kedro.framework.session.KedroSession"):
        with get_context_and_pipeline(
            CliContext("tests", Mock(package_name="tests")),
            "docker_image:latest",
            "__default__",
            "",
            is_local=False,
        ) as cx_p:
            yield cx_p


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
