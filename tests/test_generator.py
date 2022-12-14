from unittest.mock import patch

from kedro.pipeline import node, pipeline
from sagemaker.workflow import pipeline_context
from sagemaker.workflow.steps import StepTypeEnum

from kedro_sagemaker.config import _CONFIG_TEMPLATE
from kedro_sagemaker.generator import KedroSageMakerGenerator


def identity(x):
    return x


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
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=True
    )

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
    generator = KedroSageMakerGenerator(
        "__default__", context_mock, config, is_local=True
    )

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
