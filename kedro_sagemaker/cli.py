import json
import os
import subprocess
import sys
from pathlib import Path
from shlex import shlex
from typing import Optional

import click
from kedro.framework.startup import ProjectMetadata

from kedro_sagemaker.cli_functions import get_context_and_pipeline
from kedro_sagemaker.client import SageMakerClient
from kedro_sagemaker.config import CONFIG_TEMPLATE_YAML
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME,
    KEDRO_SAGEMAKER_DEBUG,
    KEDRO_SAGEMAKER_WORKING_DIRECTORY,
    KEDRO_SAGEMAKER_ARGS,
)
from kedro_sagemaker.utils import CliContext, KedroContextManager, parse_flat_parameters


@click.group("SageMaker")
def commands():
    """Kedro plugin adding support for Azure ML Pipelines"""
    pass


@commands.group(
    name="sagemaker", context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "-e",
    "--env",
    "env",
    type=str,
    default=lambda: os.environ.get("KEDRO_ENV", "local"),
    help="Environment to use.",
)
@click.pass_obj
@click.pass_context
def sagemaker_group(ctx, metadata: ProjectMetadata, env):
    click.echo(metadata)
    ctx.obj = CliContext(env, metadata)


@sagemaker_group.command()
@click.argument("bucket")
@click.argument("execution_role")
@click.argument("docker_image")
@click.pass_obj
def init(ctx: CliContext, bucket, execution_role, docker_image):
    """
    Creates basic configuration for Kedro AzureML plugin
    """
    target_path = Path.cwd().joinpath("conf/base/sagemaker.yml")
    cfg = CONFIG_TEMPLATE_YAML.format(
        **{
            "bucket": bucket,
            "execution_role": execution_role,
            "docker_image": docker_image,
        }
    )
    target_path.write_text(cfg)

    click.echo(f"Configuration generated in {target_path}")

    click.echo(
        click.style(
            f"It's recommended to perform S3 Lifecycle configuration for bucket {bucket} "
            f"to avoid costs of long-term storage of the temporary data."
            f"\nTemporary data will be stored under s3://{bucket}/{KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME} path"  # noqa
            f"\nSee https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html",  # noqa
            fg="green",
        )
    )


@sagemaker_group.command()
@click.option(
    "-r",
    "--execution_role",
    type=str,
    help="SageMaker execution role",
)
@click.option(
    "-i",
    "--image",
    type=str,
    help="Docker image to use for pipeline execution.",
)
@click.option(
    "-p",
    "--pipeline",
    "pipeline",
    type=str,
    help="Name of pipeline to run",
    default="__default__",
)
@click.option(
    "--params",
    "params",
    type=str,
    help="Parameters override in form of JSON string",
)
@click.option(
    "--local",
    is_flag=True,
    type=bool,
    default=False,
    help="If set, SageMaker LocalSession will be used to run the pipeline",
)
@click.option("--wait-for-completion", type=bool, is_flag=True, default=False)
@click.pass_obj
@click.pass_context
def run(
    click_context: click.Context,
    ctx: CliContext,
    execution_role: Optional[str],
    image: Optional[str],
    pipeline: str,
    params: str,
    wait_for_completion: bool,
    local: bool,
):
    mgr: KedroContextManager
    with get_context_and_pipeline(
        ctx, image, pipeline, params, local, execution_role
    ) as (
        mgr,
        sm_pipeline,
    ):
        client = SageMakerClient(
            sm_pipeline, execution_role or mgr.plugin_config.aws.execution_role
        )
        is_ok = client.run(
            local,
            wait_for_completion,
            lambda p: click.echo(f"Pipeline ARN: {p['PipelineArn']}"),
        )

        if is_ok:
            exit_code = 0
            click.echo(
                click.style(
                    "Pipeline {} successfully".format(
                        "finished" if wait_for_completion else "started"
                    ),
                    fg="green",
                )
            )
        else:
            exit_code = 1
            click.echo(
                click.style("There was an error while running the pipeline", fg="red")
            )

        click_context.exit(exit_code)


@sagemaker_group.command(
    hidden=True, context_settings=dict(ignore_unknown_options=True)
)
@click.pass_obj
@click.pass_context
def entrypoint(click_context: click.Context, ctx: CliContext, *args, **kwargs):
    """
    Internal entrypoint only for use with Kedro SageMaker plugin
    """
    is_debug = int(os.environ.get(KEDRO_SAGEMAKER_DEBUG, "0"))
    if is_debug:
        print("\n".join(sys.argv[1:]))
        print("-" * 80)
        print(json.dumps(dict(os.environ), indent=4))

    args = os.environ.get(KEDRO_SAGEMAKER_ARGS)

    os.chdir(os.environ[KEDRO_SAGEMAKER_WORKING_DIRECTORY])
    kedro_params = parse_flat_parameters(os.environ)

    result = subprocess.run(
        shlex.join(shlex.split(args) + ["--params", json.dumps(kedro_params)]),
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        shell=True,
        check=False,
    )

    click_context.exit(result.returncode)
