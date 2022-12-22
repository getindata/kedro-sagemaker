from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline

from kedro_sagemaker.runner import SageMakerPipelinesRunner


def test_can_invoke_dummy_pipeline(
    dummy_pipeline: Pipeline, patched_sagemaker_runner: SageMakerPipelinesRunner
):
    runner = patched_sagemaker_runner
    catalog = DataCatalog()
    input_data = ["yolo :)"]
    catalog.add("input_data", MemoryDataSet(data=input_data))
    results = runner.run(
        dummy_pipeline,
        catalog,
    )
    assert results["output_data"] == input_data, "No output data found"
    assert bool(runner.runner_name()), "Name not returned"


def test_runner_fills_missing_datasets(
    dummy_pipeline: Pipeline, patched_sagemaker_runner: SageMakerPipelinesRunner
):
    input_data = ["yolo :)"]
    runner = patched_sagemaker_runner
    catalog = DataCatalog()
    catalog.add("input_data", MemoryDataSet(data=input_data))
    for node_no in range(3):
        results = runner.run(
            dummy_pipeline.filter(node_names=[f"node{node_no+1}"]),
            catalog,
        )
    assert results["output_data"] == input_data, "Invalid output data"
