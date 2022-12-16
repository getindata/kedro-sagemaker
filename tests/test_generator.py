import json
from unittest.mock import patch

from kedro.io import DataCatalog
from kedro.pipeline import node, pipeline
from sagemaker.workflow import pipeline_context
from sagemaker.workflow.steps import StepTypeEnum

from kedro_sagemaker.config import _CONFIG_TEMPLATE, ResourceConfig
from kedro_sagemaker.datasets import SageMakerModelDataset
from kedro_sagemaker.decorators import sagemaker_metrics
from kedro_sagemaker.generator import KedroSageMakerGenerator
from tests.utils import identity

sample_pipeline = pipeline(
    [
        node(identity, inputs="input_data", outputs="i2", name="node1"),
        node(identity, inputs="i2", outputs="output_data", name="node2"),
    ]
)


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_generate_pipeline_with_processing_steps(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=False
    )

    # when
    pipeline = generator.generate()

    # then
    assert isinstance(pipeline.sagemaker_session, pipeline_context.PipelineSession)
    assert pipeline.name == "kedro-sagemaker-pipeline"  # a default one
    assert len(pipeline.parameters) == 0
    assert len(pipeline.steps) == 2

    assert pipeline.steps[0].name == "node1"
    assert pipeline.steps[0].step_type == StepTypeEnum.PROCESSING
    assert pipeline.steps[0].depends_on is None

    assert pipeline.steps[1].name == "node2"
    assert pipeline.steps[1].step_type == StepTypeEnum.PROCESSING
    assert pipeline.steps[1].depends_on[0].name == "node1"


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_generate_local_pipeline(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=True
    )

    # when
    pipeline = generator.generate()

    # then
    assert isinstance(pipeline.sagemaker_session, pipeline_context.LocalPipelineSession)


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_use_mapping_for_pipeline_name(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    config.aws.sagemaker.pipeline_names_mapping = {"__default__": "training"}
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    # when
    pipeline = generator.generate()

    # then
    assert pipeline.name == "training"


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_process_kedro_parameters(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    context_mock.params = {
        "numeric": {"int_param": 42, "float_param": 3.14},
        "string": "alamakota",
        "is_great_plugin": True,
        "features": ["age", "gender"],
    }
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    # when
    pipeline = generator.generate()

    # then
    assert len(pipeline.parameters) == 10
    params_dict = {param.name: param for param in pipeline.parameters}
    params_transformed = {
        params_dict[f"param{i}"].default_value: params_dict[f"value{i}"]
        for i in range(5)
    }

    assert params_transformed["numeric.int_param"].parameter_type.python_type == int
    assert params_transformed["numeric.int_param"].default_value == 42
    assert params_transformed["numeric.float_param"].parameter_type.python_type == float
    assert params_transformed["numeric.float_param"].default_value == 3.14
    assert params_transformed["string"].parameter_type.python_type == str
    assert params_transformed["string"].default_value == "alamakota"
    assert params_transformed["is_great_plugin"].parameter_type.python_type == bool
    assert params_transformed["is_great_plugin"].default_value is True
    assert params_transformed["features"].parameter_type.python_type == str
    assert params_transformed["features"].default_value == '["age", "gender"]'


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_create_processor_based_on_the_config(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    config.aws.execution_role = "__execution_role__"
    config.aws.bucket = "__bucket_name__"
    config.docker.image = "__image_uri__"
    config.aws.resources["node1"] = ResourceConfig(
        instance_type="__instance_type__", instance_count=42, timeout_seconds=4242
    )
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    # when
    pipeline = generator.generate()

    # then
    processor = pipeline.steps[0].processor
    assert processor.entrypoint == [
        "kedro",
        "sagemaker",
        "execute",
        "--pipeline=__default__",
        "--node=node1",
    ]
    assert processor.role == "__execution_role__"
    assert processor.image_uri == "__image_uri__"
    assert processor.instance_count == 42
    assert processor.instance_type == "__instance_type__"
    assert processor.max_runtime_in_seconds == 4242
    assert (
        json.loads(processor.env["KEDRO_SAGEMAKER_RUNNER_CONFIG"])["bucket"]
        == "__bucket_name__"
    )
    assert "run_id" in json.loads(processor.env["KEDRO_SAGEMAKER_RUNNER_CONFIG"])


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_use_default_resources_spec_in_processing_step(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    config.aws.resources["__default__"] = ResourceConfig(
        instance_type="__default_instance_type__",
        instance_count=142,
        timeout_seconds=14242,
    )
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    # when
    pipeline = generator.generate()

    # then
    processor = pipeline.steps[0].processor
    assert processor.instance_count == 142
    assert processor.instance_type == "__default_instance_type__"
    assert processor.max_runtime_in_seconds == 14242


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
@patch("kedro_sagemaker.generator.Model")
@patch("kedro_sagemaker.generator.ModelStep")
def test_should_generate_training_steps_and_register_model(
    model_step_mock, model_mock, context_mock
):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    config.docker.image = "__image_uri__"
    context_mock.catalog = DataCatalog({"i2": SageMakerModelDataset()})
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=False
    )

    # when
    pipeline = generator.generate()

    # then
    assert len(pipeline.steps) == 3
    assert pipeline.steps[1].step_type == StepTypeEnum.TRAINING
    assert pipeline.steps[2].step_type == StepTypeEnum.PROCESSING
    model_properties = model_mock.call_args.kwargs
    assert model_properties["name"] == "i2"
    assert model_properties["image_uri"] == "__image_uri__"
    assert model_step_mock.call_args_list[0].args[0] == "CreateModel0"
    assert model_step_mock.call_args_list[1].args[0] == "RegisterModel0"


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
@patch("kedro_sagemaker.generator.Model")
@patch("kedro_sagemaker.generator.ModelStep")
def test_should_generate_training_steps_and_skip_model_registration(
    model_step_mock, model_mock, context_mock
):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    context_mock.catalog = DataCatalog({"i2": SageMakerModelDataset()})
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=True
    )

    # when
    generator.generate()

    # then
    assert len(model_step_mock.call_args_list) == 1
    assert model_step_mock.call_args_list[0].args[0] == "CreateModel0"
    # no model registration


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
@patch("kedro_sagemaker.generator.Model")
@patch("kedro_sagemaker.generator.ModelStep")
def test_should_create_estimator_based_on_the_config(
    model_step_mock, model_mock, context_mock
):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    config.aws.execution_role = "__execution_role__"
    config.aws.bucket = "__bucket_name__"
    config.docker.image = "__image_uri__"
    config.aws.resources["node1"] = ResourceConfig(
        instance_type="__instance_type__", instance_count=42, timeout_seconds=4242
    )
    context_mock.catalog = DataCatalog({"i2": SageMakerModelDataset()})
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    # when
    pipeline = generator.generate()

    # then
    estimator = pipeline.steps[1].estimator
    assert estimator.image_uri == "__image_uri__"
    assert estimator.role == "__execution_role__"
    assert estimator.instance_count == 42
    assert estimator.instance_type == "__instance_type__"
    assert estimator.max_run == 4242
    assert estimator.enable_sagemaker_metrics is False
    assert len(estimator.metric_definitions) == 0
    assert (
        estimator.environment["KEDRO_SAGEMAKER_ARGS"]
        == "kedro sagemaker execute --pipeline=__default__ --node=node1"
    )
    assert estimator.environment["KEDRO_SAGEMAKER_WD"] == "/home/kedro"
    assert (
        json.loads(estimator.environment["KEDRO_SAGEMAKER_RUNNER_CONFIG"])["bucket"]
        == "__bucket_name__"
    )
    assert "run_id" in json.loads(
        estimator.environment["KEDRO_SAGEMAKER_RUNNER_CONFIG"]
    )


@patch("kedro.framework.project.pipelines", {"__default__": sample_pipeline})
@patch("kedro.framework.context.KedroContext")
def test_should_mark_node_as_estimator_if_it_exposes_metrics(context_mock):
    # given
    config = _CONFIG_TEMPLATE.copy(deep=True)
    generator = KedroSageMakerGenerator("__default__", context_mock, config)

    try:

        @sagemaker_metrics({"auc": "AUC: .*"})
        def identity_with_metric(x):
            pass

        sample_pipeline.nodes[0].func = identity_with_metric

        # when
        pipeline = generator.generate()

        # then
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].step_type == StepTypeEnum.TRAINING
        assert pipeline.steps[1].step_type == StepTypeEnum.PROCESSING
        estimator = pipeline.steps[0].estimator
        assert estimator.enable_sagemaker_metrics is True
        assert estimator.metric_definitions == [{"Name": "auc", "Regex": "AUC: .*"}]
    finally:
        sample_pipeline.nodes[0].func = identity
