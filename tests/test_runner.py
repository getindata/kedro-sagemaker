import os
from unittest.mock import patch

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline

from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_EXECUTION_ARN,
    KEDRO_SAGEMAKER_RUNNER_CONFIG,
)
from kedro_sagemaker.runner import SageMakerPipelinesRunner


def test_can_invoke_dummy_pipeline(
    dummy_pipeline: Pipeline, patched_sagemaker_runner: SageMakerPipelinesRunner
):
    runner = patched_sagemaker_runner
    catalog = DataCatalog()
    input_data = ["yolo :)"]
    catalog.add("input_data", MemoryDataset(data=input_data))
    results = runner.run(
        dummy_pipeline,
        catalog,
    )
    assert results["output_data"] == input_data, "No output data found"
    assert bool(runner.runner_name()), "Name not returned"


# def test_runner_fills_missing_datasets(
#     dummy_pipeline: Pipeline, patched_sagemaker_runner: SageMakerPipelinesRunner
# ):
#     input_data = ["yolo :)"]
#     runner = patched_sagemaker_runner
#     catalog = DataCatalog()
#     catalog.add("input_data", MemoryDataset(data=input_data))
#     for node_no in range(3):
#         results = runner.run(
#             dummy_pipeline.filter(node_names=[f"node{node_no+1}"]),
#             catalog,
#         )
#     assert results["output_data"] == input_data, "Invalid output data"


def test_runner_creating_default_datasets_based_on_execution_arn():
    with patch.dict(
        os.environ,
        {
            KEDRO_SAGEMAKER_EXECUTION_ARN: "execution-arn",
            KEDRO_SAGEMAKER_RUNNER_CONFIG: '{"bucket": "s3-bucket"}',
        },
    ):
        runner = SageMakerPipelinesRunner()
        dataset = runner.create_default_data_set("output_data")
        assert (
            dataset._get_target_path()
            == "s3://s3-bucket/kedro-sagemaker-tmp/execution-arn/output_data.bin"
        )
