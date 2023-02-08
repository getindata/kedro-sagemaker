import importlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import click
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

from kedro_sagemaker.constants import MLFLOW_TAG_EXECUTION_ARN
from kedro_sagemaker.generator import KedroSageMakerGenerator
from kedro_sagemaker.utils import (
    CliContext,
    KedroContextManager,
    docker_build,
    docker_push,
)

logger = logging.getLogger()


def parse_extra_params(params, silent=False):
    if params and (parameters := json.loads(params.strip("'"))):
        if not silent:
            click.echo(
                f"Running with extra parameters:\n{json.dumps(parameters, indent=4)}"
            )
    else:
        parameters = None
    return parameters


@contextmanager
def get_context_and_pipeline(
    ctx: CliContext,
    docker_image: Optional[str],
    pipeline: str,
    params: str,
    is_local: bool,
    execution_role_arn: Optional[str] = None,
) -> Iterator[Tuple[KedroContextManager, SageMakerPipeline]]:
    with KedroContextManager(
        ctx.metadata.package_name, ctx.env, parse_extra_params(params, True)
    ) as mgr:
        generator = KedroSageMakerGenerator(
            pipeline,
            mgr.context,
            mgr.plugin_config,
            docker_image,
            is_local,
            execution_role_arn,
        )
        sm_pipeline = generator.generate()
        yield mgr, sm_pipeline


def docker_autobuild(auto_build, click_context, image, mgr, yes):
    if auto_build:
        if (splits := image.split(":"))[-1] != "latest" and len(splits) > 1:
            click.echo(
                click.style(
                    f"This operation will overwrite the target image with {splits[-1]} tag at remote location.",
                    fg="yellow",
                )
            )

        if not yes and not click.confirm("Continue?", default=True):
            click_context.exit(1)

        if (rv := docker_build(str(mgr.context.project_path), image)) != 0:
            click_context.exit(rv)
        if (rv := docker_push(image)) != 0:
            click_context.exit(rv)
    else:
        click.echo(
            click.style(
                "Make sure that you've built and pushed your image to run the latest version remotely.\
 Consider using '--auto-build' parameter.",
                fg="yellow",
            )
        )


def write_file_and_confirm_overwrite(
    filepath: Path,
    auto_overwrite: bool,
    contents: str,
    on_denied_overwrite: Callable[[Path], None] = None,
):
    write = True
    if filepath.exists():
        write = False
        if auto_overwrite or click.confirm(
            f"{filepath} exists, overwrite?", default=True
        ):
            write = True
    if write:
        filepath.write_text(contents)
    elif on_denied_overwrite:
        on_denied_overwrite(filepath)


def lookup_mlflow_run_id(context, sagemaker_execution_arn: str):
    import mlflow
    from kedro_mlflow.config.kedro_mlflow_config import KedroMlflowConfig

    mlflow_conf: KedroMlflowConfig = context.mlflow
    mlflow_runs = mlflow.search_runs(
        experiment_names=[mlflow_conf.tracking.experiment.name],
        filter_string=f'tags.`{MLFLOW_TAG_EXECUTION_ARN}` = "{sagemaker_execution_arn}"',
        max_results=1,
        output_format="list",
    )
    importlib.reload(mlflow.tracking.request_header.registry)

    if len(mlflow_runs) == 0:
        logger.warn(
            "Unable to find parent mlflow run id for the current execution (%s)",
            sagemaker_execution_arn,
        )
        return mlflow.tracking._RUN_ID_ENV_VAR, None

    return mlflow.tracking._RUN_ID_ENV_VAR, mlflow_runs[0].info.run_id
