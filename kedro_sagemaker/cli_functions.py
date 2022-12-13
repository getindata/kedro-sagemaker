import json
import os
from contextlib import contextmanager
from typing import Optional, Iterator, Tuple
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
import click

from kedro_sagemaker.generator import KedroSageMakerGenerator
from kedro_sagemaker.utils import (
    KedroContextManager,
    CliContext,
    docker_build,
    docker_push,
)


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
        sm_pipeline.describe()
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
