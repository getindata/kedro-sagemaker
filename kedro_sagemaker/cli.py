import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from kedro.framework.startup import ProjectMetadata

from kedro_sagemaker.cli_functions import (
    docker_autobuild,
    get_context_and_pipeline,
    parse_extra_params,
    write_file_and_confirm_overwrite,
)
from kedro_sagemaker.client import SageMakerClient
from kedro_sagemaker.config import CONFIG_TEMPLATE_YAML
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_ARGS,
    KEDRO_SAGEMAKER_DEBUG,
    KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME,
    KEDRO_SAGEMAKER_WORKING_DIRECTORY,
)
from kedro_sagemaker.docker import DOCKERFILE_TEMPLATE, DOCKERIGNORE_TEMPLATE
from kedro_sagemaker.runner import SageMakerPipelinesRunner
from kedro_sagemaker.utils import (
    CliContext,
    KedroContextManager,
    parse_flat_parameters,
)


@click.group("SageMaker")
def commands():
    """Kedro plugin adding support for SageMaker Pipelines"""
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
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    type=bool,
    help="Auto answer yes confirm prompts",
)
@click.pass_obj
def init(ctx: CliContext, bucket, execution_role, docker_image, yes: bool):
    """
    Creates basic configuration for Kedro AzureML plugin
    """
    cwd = Path.cwd()
    target_path = cwd.joinpath("conf/base/sagemaker.yml")
    cfg = CONFIG_TEMPLATE_YAML.format(
        **{
            "bucket": bucket,
            "execution_role": execution_role,
            "docker_image": docker_image,
        }
    )
    target_path.write_text(cfg)

    click.echo(f"Configuration generated in {target_path}")

    def on_denied_overwrite(filepath: Path):
        click.echo(
            click.style(
                f"Kedro SageMaker recommends to use auto-generated {filepath.name} "
                "to ensure SageMaker-compatible docker images",
                fg="yellow",
            )
        )

    write_file_and_confirm_overwrite(
        cwd / "Dockerfile", yes, DOCKERFILE_TEMPLATE, on_denied_overwrite
    )
    write_file_and_confirm_overwrite(
        cwd / ".dockerignore", yes, DOCKERIGNORE_TEMPLATE, on_denied_overwrite
    )

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
    "--execution-role",
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
@click.option("--wait-for-completion", type=bool, is_flag=True, default=False)
@click.option(
    "--local",
    is_flag=True,
    type=bool,
    default=False,
    help="If set, SageMaker LocalSession will be used to run the pipeline",
)
@click.option(
    "--auto-build",
    type=bool,
    is_flag=True,
    default=False,
    help="Specify to docker build and push before scheduling a run",
)
@click.option(
    "--yes",
    "-y",
    type=bool,
    is_flag=True,
    default=False,
    help="Auto answer yes confirm prompts",
)
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
    auto_build: bool,
    yes: bool,
):
    """
    Runs the pipeline on SageMaker Pipelines
    """
    mgr: KedroContextManager
    with get_context_and_pipeline(
        ctx, image, pipeline, params, local, execution_role
    ) as (
        mgr,
        sm_pipeline,
    ):
        docker_autobuild(
            auto_build, click_context, image or mgr.plugin_config.docker.image, mgr, yes
        )

        client = SageMakerClient(
            sm_pipeline, execution_role or mgr.plugin_config.aws.execution_role
        )

        is_ok = client.run(
            local,
            wait_for_completion,
            lambda p: click.echo(f"Pipeline ARN: {p.describe()['PipelineArn']}"),
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


@sagemaker_group.command()
@click.option(
    "-r",
    "--execution-role",
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
@click.option(
    "--output-path",
    type=click.types.Path(exists=False, dir_okay=False),
    default="pipeline.json",
    help="Path to output pipeline JSON file. Defaults to ./pipeline.json",
)
@click.pass_obj
def compile(
    ctx: CliContext,
    execution_role: Optional[str],
    image: Optional[str],
    pipeline: str,
    params: str,
    local: bool,
    output_path: str,
):
    """
    Compiles the pipeline to a JSON file
    """
    with get_context_and_pipeline(
        ctx, image, pipeline, params, local, execution_role
    ) as (
        _,
        sm_pipeline,
    ):
        target_path = Path(output_path)
        with target_path.open("w") as f:
            json.dump(json.loads(sm_pipeline.definition()), f, indent=4)
        click.echo(f"Pipeline compiled to {target_path}")


@sagemaker_group.command(
    hidden=True,
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.pass_obj
@click.pass_context
def entrypoint(click_context: click.Context, ctx: CliContext, *other_args, **kwargs):
    """
    Internal entrypoint only for use with Kedro SageMaker plugin
    """
    is_debug = int(os.environ.get(KEDRO_SAGEMAKER_DEBUG, "0"))
    if is_debug:
        print("\n".join(sys.argv[1:]))
        print("-" * 80)
        print(json.dumps(dict(os.environ), indent=4))

    args = os.environ.get(KEDRO_SAGEMAKER_ARGS)

    os.chdir(os.environ.get(KEDRO_SAGEMAKER_WORKING_DIRECTORY))
    kedro_params = parse_flat_parameters(os.environ)

    result = subprocess.run(
        shlex.join(shlex.split(args) + ["--params", json.dumps(kedro_params)]),
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        shell=True,
        check=False,
    )

    click_context.exit(result.returncode)


@sagemaker_group.command(hidden=True)
@click.option(
    "-p",
    "--pipeline",
    "pipeline",
    type=str,
    help="Name of pipeline to run",
    default="__default__",
)
@click.option(
    "-n", "--node", "node", type=str, help="Name of the node to run", required=True
)
@click.option(
    "--params",
    "params",
    type=str,
    help="Parameters override in form of JSON string",
)
@click.pass_obj
def execute(ctx: CliContext, pipeline: str, node: str, params: str):
    parameters = parse_extra_params(params)
    with KedroContextManager(
        ctx.metadata.package_name, env=ctx.env, extra_params=parameters
    ) as mgr:
        runner = SageMakerPipelinesRunner()
        mgr.session.run(pipeline, node_names=[node], runner=runner)
