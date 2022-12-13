import json
from itertools import chain
from typing import (
    List,
    Union,
    Tuple,
    Any,
    Dict,
    Optional,
    Iterator,
)
from uuid import uuid4

from kedro.framework.context import KedroContext
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline as KedroPipeline
from kedro.pipeline.node import Node as KedroNode
from sagemaker import Processor, Model
from sagemaker.estimator import Estimator
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from kedro_sagemaker.config import (
    SageMakerMetricsTrackingConfig,
    ResourceConfig,
    KedroSageMakerPluginConfig,
)
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_METRICS,
    KEDRO_SAGEMAKER_ARGS,
    KEDRO_SAGEMAKER_WORKING_DIRECTORY,
    KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
    KEDRO_SAGEMAKER_RUNNER_CONFIG,
)
from kedro_sagemaker.datasets import SageMakerModelDataset
from kedro_sagemaker.runner import (
    SageMakerPipelinesRunner,
    KedroSageMakerRunnerConfig,
)
from kedro_sagemaker.utils import flatten_dict

SageMakerStepType = Union[ProcessingStep, TrainingStep, ModelStep]


class KedroSageMakerGenerator:
    def __init__(
        self,
        pipeline_name: str,
        kedro_context: KedroContext,
        config: KedroSageMakerPluginConfig,
        docker_image: Optional[str] = None,
        is_local: bool = False,
        execution_role_arn: Optional[str] = None,
    ):
        self.is_local = is_local
        self.docker_image = docker_image
        self.config = config
        self.kedro_context = kedro_context
        self.pipeline_name = pipeline_name
        self.execution_role_arn = execution_role_arn

    @property
    def _execution_role(self):
        return self.execution_role_arn or self.config.aws.execution_role

    def _datasets_of_type(
        self, t: type, inputs_or_outputs: List[str], catalog: DataCatalog
    ) -> Iterator[str]:
        defined_datasets = set(catalog.list())
        for ds in inputs_or_outputs:
            if ds in defined_datasets and isinstance(catalog._get_dataset(ds), t):
                yield ds

    def _prepare_sagemaker_params(self) -> Tuple[list, dict]:
        sm_param_types = {
            int: ParameterInteger,
            float: ParameterFloat,
            bool: ParameterBoolean,
            str: ParameterString,
        }

        sm_param_envs = {}
        sm_kedro_params = []
        for node_idx, (k, v) in enumerate(
            flatten_dict(self.kedro_context.params).items()
        ):
            sm_param_key = ParameterString(f"param{node_idx}", default_value=k)
            value_name = f"value{node_idx}"

            if (t := type(v)) in sm_param_types:
                sm_param_value = sm_param_types[t](value_name, default_value=v)
            else:
                sm_param_value = ParameterString(
                    value_name, default_value=json.dumps(v)
                )

            sm_kedro_params.append(sm_param_key)
            sm_kedro_params.append(sm_param_value)

            sm_param_envs[
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + str(node_idx)
            ] = sm_param_key
            sm_param_envs[KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + str(node_idx)] = (
                sm_param_value
                if isinstance(sm_param_value, ParameterString)
                else sm_param_value.to_string()
            )
        return sm_kedro_params, sm_param_envs

    def get_kedro_pipeline(self) -> KedroPipeline:
        from kedro.framework.project import pipelines

        pipeline: KedroPipeline = pipelines[self.pipeline_name]
        return pipeline

    def _get_sagemaker_pipeline_name(self) -> str:
        return (self.config.aws.sagemaker.pipeline_names_mapping or {}).get(
            self.pipeline_name, "kedro-sagemaker-pipeline"
        )

    def _get_default_resources(self) -> ResourceConfig:
        return ResourceConfig(instance_type="ml.m5.large")

    def _get_resources_for_node(self, node: KedroNode):
        node_resources = next(
            (
                self.config.aws.resources.get(n)
                for n in chain([node.name], iter(node.tags))
            ),
            None,
        )

        defined_default = self.config.aws.resources.get("__default__", None)
        defaults = self._get_default_resources()
        if defined_default:
            defaults = defaults.copy(update=defined_default.dict())

        if node_resources:
            return defaults.copy(update=node_resources.dict())
        else:
            return defaults

    def generate(self) -> SageMakerPipeline:
        run_id = uuid4().hex
        runner_config = KedroSageMakerRunnerConfig(
            bucket=self.config.aws.bucket, run_id=run_id
        )

        sagemaker_session = (
            LocalPipelineSession() if self.is_local else PipelineSession()
        )

        sm_params = []

        sm_params_kedro, sm_param_envs = self._prepare_sagemaker_params()
        sm_params.extend(sm_params_kedro)

        steps: Dict[str, SageMakerStepType] = {}
        sm_training_steps_with_model_outputs = {}

        pipeline = self.get_kedro_pipeline()

        for node_idx, node in enumerate(pipeline.nodes):
            node_resources = self._get_resources_for_node(node)
            sm_node_name = node.name.replace(".", "__")

            sagemaker_model_inputs, sagemaker_model_outputs = self._get_model_io(node)

            has_sagemaker_model_output = len(sagemaker_model_outputs) > 0

            is_training_node = (
                hasattr(node.func, KEDRO_SAGEMAKER_METRICS)
                or has_sagemaker_model_output
            )

            if is_training_node:
                sm_metrics = getattr(
                    node.func,
                    KEDRO_SAGEMAKER_METRICS,
                    SageMakerMetricsTrackingConfig({}),
                )

                step = self._create_training_step(
                    node,
                    node_resources,
                    runner_config,
                    sagemaker_model_inputs,
                    sm_metrics,
                    sm_node_name,
                    sm_param_envs,
                    sm_training_steps_with_model_outputs,
                )

                assert (
                    len(sagemaker_model_outputs) <= 1
                ), "There can only be 1 SageMaker Model output per Kedro node!"
                if has_sagemaker_model_output:
                    model_steps = self._create_model_register_steps(
                        node_idx,
                        sagemaker_model_outputs,
                        sagemaker_session,
                        sm_training_steps_with_model_outputs,
                        step,
                    )
                    for ms in model_steps:
                        steps[f"{node.name}-{ms.name}"] = ms

            else:
                step = self._create_processing_step(
                    node,
                    node_resources,
                    runner_config,
                    sm_node_name,
                    sm_param_envs,
                )
            steps[node.name] = step

        steps = self._add_step_dependencies(pipeline, steps)

        smp = SageMakerPipeline(
            self._get_sagemaker_pipeline_name(),
            sagemaker_session=sagemaker_session,
            steps=list(steps.values()),
            parameters=sm_params,
        )
        return smp

    def _add_step_dependencies(
        self, pipeline, steps: Dict[str, SageMakerStepType]
    ) -> Dict[str, SageMakerStepType]:
        for node, dependencies in pipeline.node_dependencies.items():
            deps_names = set(d.name for d in dependencies)
            sm_step = steps[node.name]
            sm_deps = [s for n, s in steps.items() if n in deps_names]
            sm_step.add_depends_on(sm_deps)
        return steps

    def _create_processing_step(
        self, node, node_resources, runner_config, sm_node_name, sm_param_envs
    ):
        step = ProcessingStep(
            sm_node_name,
            processor=Processor(
                entrypoint=self._get_kedro_command(node),
                role=self._execution_role,
                image_uri=self.config.docker.image,
                instance_count=node_resources.instance_count,
                instance_type=node_resources.instance_type,
                max_runtime_in_seconds=node_resources.timeout_seconds,
                env={
                    KEDRO_SAGEMAKER_RUNNER_CONFIG: runner_config.json(),
                    **sm_param_envs,
                },
            ),
            display_name=node.name,
            description=node.name,
        )
        return step

    def _get_kedro_command(self, node, as_string=False) -> Union[str, List[str]]:
        cmd = [
            "kedro",
            "sagemaker",
            "execute",
            f"--pipeline={self.pipeline_name}",
            f"--node={node.name}",
        ]
        if as_string:
            return " ".join(cmd)
        else:
            return cmd

    def _create_model_register_steps(
        self,
        node_idx,
        sagemaker_model_outputs,
        sagemaker_session,
        sm_training_steps_with_model_outputs,
        step,
    ) -> Union[Tuple[ModelStep], Tuple[ModelStep, ModelStep]]:
        model_output_name = sagemaker_model_outputs[0].replace(".", "__")
        model = Model(
            image_uri=self.config.docker.image,
            model_data=step.properties.ModelArtifacts.S3ModelArtifacts,
            role=self._execution_role,
            name=model_output_name,
            sagemaker_session=sagemaker_session,
        )
        step_model_create = ModelStep(
            f"CreateModel{node_idx}",
            step_args=model.create(),
        )
        sm_training_steps_with_model_outputs[sagemaker_model_outputs[0]] = step
        results = (step_model_create,)

        if not self.is_local:
            model_registry_kwargs = dict(
                model_package_group_name="KedroSageMakerModels",
                content_types=["application/json"],
                response_types=["application/json"],
                domain="MACHINE_LEARNING",
                task="OTHER",
                image_uri=self.config.docker.image,
            )

            # TODO - maybe model metrics from https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html ?
            ds: SageMakerModelDataset = self.kedro_context.catalog._get_dataset(
                sagemaker_model_outputs[0]
            )  # noqa

            model_registry_kwargs.update(ds.model_registry_kwargs)

            step_model_register = ModelStep(
                f"RegisterModel{node_idx}",
                step_args=model.register(**model_registry_kwargs),
            )

            results = results + (step_model_register,)
        return results

    def _create_training_step(
        self,
        node,
        node_resources,
        runner_config,
        sagemaker_model_inputs,
        sm_metrics,
        sm_node_name,
        sm_param_envs,
        sm_training_steps_with_model_outputs,
    ):
        return TrainingStep(
            sm_node_name,
            estimator=Estimator(
                image_uri=self.config.docker.image,
                role=self._execution_role,
                instance_count=node_resources.instance_count,
                instance_type=node_resources.instance_type,
                max_run=node_resources.timeout_seconds,
                metric_definitions=[
                    {"Name": k, "Regex": v} for k, v in sm_metrics.metrics.items()
                ],
                enable_sagemaker_metrics=len(sm_metrics.metrics) > 0,
                environment={
                    KEDRO_SAGEMAKER_ARGS: self._get_kedro_command(node, as_string=True),
                    KEDRO_SAGEMAKER_RUNNER_CONFIG: runner_config.json(),
                    KEDRO_SAGEMAKER_WORKING_DIRECTORY: self.config.docker.working_directory,
                    # "PYTHONPATH": "/home/kedro/src",
                    # # TODO - this will not be needed if plugin is installed, I hope :D,
                    **sm_param_envs,
                },
                base_job_name=sm_node_name,
                model_uri=sm_training_steps_with_model_outputs[
                    sagemaker_model_inputs[0]
                ].properties.ModelArtifacts.S3ModelArtifacts
                if sagemaker_model_inputs
                else None,
            ),
            display_name=node.name,
            description=node.name,
        )

    def _get_model_io(self, node):
        sagemaker_model_inputs = list(
            self._datasets_of_type(
                SageMakerModelDataset, node.inputs, self.kedro_context.catalog
            )
        )
        sagemaker_model_outputs = list(
            self._datasets_of_type(
                SageMakerModelDataset, node.outputs, self.kedro_context.catalog
            )
        )
        return sagemaker_model_inputs, sagemaker_model_outputs
