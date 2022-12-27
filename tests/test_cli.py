import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
import yaml
from click.testing import CliRunner
from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

from kedro_sagemaker import cli
from kedro_sagemaker.config import KedroSageMakerPluginConfig
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_ARGS,
    KEDRO_SAGEMAKER_DEBUG,
    KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
    KEDRO_SAGEMAKER_WORKING_DIRECTORY,
)
from kedro_sagemaker.generator import KedroSageMakerGenerator
from tests.utils import assert_has_any_call_with_args


@patch("click.confirm", return_value=True)
@pytest.mark.parametrize(
    "clean_dir", (False, True), ids=("with existing config", "empty dir")
)
@pytest.mark.parametrize("yes", (False, True), ids=("without --yes", "with --yes"))
def test_can_initialize_basic_plugin_config(
    click_confirm,
    clean_dir,
    yes,
    patched_kedro_package,
    cli_context,
):
    # given
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("conf/base").mkdir(parents=True)

        if not clean_dir:
            (Path.cwd() / "Dockerfile").write_text("FROM python:3.10")
            (Path.cwd() / ".dockerignore").write_text("")

        # when
        result = runner.invoke(
            cli.init,
            [
                "__bucket_name__",
                "__execution_role__",
                "__docker_image__",
            ]
            + (["--yes"] if yes else []),
            obj=cli_context,
        )

        # then
        assert result.exit_code == 0

        config_path = Path("conf/base/sagemaker.yml")
        assert config_path.exists()
        config = KedroSageMakerPluginConfig.parse_obj(
            yaml.safe_load(config_path.read_text())
        )
        assert config.aws.execution_role == "__execution_role__"
        assert config.aws.bucket == "__bucket_name__"
        assert config.docker.image == "__docker_image__"

        if not yes and not clean_dir:
            click_confirm.assert_called()
        else:
            click_confirm.assert_not_called()


@pytest.mark.parametrize(
    "extra_params",
    ["", json.dumps({"my_extra_param": 123, "my": {"nested": {"parameter": 66.6}}})],
    ids=("without extra params", "with extra params"),
)
def test_can_compile_the_pipeline(
    extra_params, patched_kedro_package, cli_context, dummy_pipeline, tmp_path: Path
):
    runner = CliRunner()
    with patch.object(
        KedroSageMakerGenerator, "get_kedro_pipeline", return_value=dummy_pipeline
    ):
        output_path = tmp_path / "pipeline.json"
        result = runner.invoke(
            cli.compile,
            ["--output-path", str(output_path.absolute())]
            + (["--params", extra_params] if extra_params else []),
            obj=cli_context,
        )

        assert result.exit_code == 0
        assert isinstance(json.loads(output_path.read_text()), dict)


@patch("click.confirm")
@patch("subprocess.run", return_value=Mock(returncode=0))
@patch("kedro_sagemaker.client.SageMakerClient")
@pytest.mark.parametrize(
    "wait_for_completion", (False, True), ids=("no wait", "wait for completion")
)
@pytest.mark.parametrize(
    "extra_params",
    ["", json.dumps({"my_extra_param": 123, "my": {"nested": {"parameter": 66.6}}})],
    ids=("without extra params", "with extra params"),
)
@pytest.mark.parametrize(
    "auto_build", (False, True), ids=("no auto-build", "with auto-build")
)
@pytest.mark.parametrize("yes", (False, True), ids=("without --yes", "with --yes"))
def test_can_run_the_pipeline(
    sagemaker_client,
    subprocess_run,
    click_confirm,
    auto_build,
    extra_params,
    patched_kedro_package,
    cli_context,
    dummy_pipeline,
    yes: bool,
    tmp_path: Path,
    wait_for_completion: bool,
):
    mock_image = f"docker_image:{uuid4().hex}"
    started_pipeline = MagicMock()
    with patch.object(
        KedroSageMakerGenerator, "get_kedro_pipeline", return_value=dummy_pipeline
    ), patch.object(SageMakerPipeline, "upsert") as upsert, patch.object(
        SageMakerPipeline, "start", return_value=started_pipeline
    ) as start, patch(
        "sagemaker.model.Model"
    ), patch(
        "sagemaker.workflow.model_step.ModelStep"
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli.run,
            (["--params", extra_params] if extra_params else [])
            + (["--auto-build"] if auto_build else [])
            + (["--yes"] if yes else [])
            + (["--wait-for-completion"] if wait_for_completion else [])
            + ["-i", mock_image],
            obj=cli_context,
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        sagemaker_client.run.asset_called_once()
        upsert.assert_called_once()
        start.assert_called_once()

        assert_docker_build = lambda: assert_has_any_call_with_args(  # noqa: E731
            subprocess_run,
            ["docker", "build", str(Path.cwd().absolute()), "-t", mock_image],
        )

        assert_docker_push = lambda: assert_has_any_call_with_args(  # noqa: E731
            subprocess_run, ["docker", "push", mock_image]
        )  # noqa: E731

        if auto_build:
            assert_docker_build()
            assert_docker_push()
            if yes:
                click_confirm.assert_not_called()
            else:
                click_confirm.assert_called_once()
        else:
            with pytest.raises(AssertionError):
                assert_docker_build()
            with pytest.raises(AssertionError):
                assert_docker_push()

        if wait_for_completion:
            started_pipeline.wait.assert_called_once()
        else:
            started_pipeline.wait.assert_not_called()


@pytest.mark.parametrize("kedro_sagemaker_debug", ("0", "1"))
@patch("subprocess.run", return_value=Mock(returncode=0))
def test_sagemaker_entrypoint_can_be_called_with_any_cli_args(
    subprocess_run, tmp_path: Path, kedro_sagemaker_debug: str
):
    runner = CliRunner()
    cmd = "echo 'Unit tests'"
    with patch.dict(
        os.environ,
        {
            KEDRO_SAGEMAKER_ARGS: cmd,
            KEDRO_SAGEMAKER_WORKING_DIRECTORY: str(tmp_path.absolute()),
            KEDRO_SAGEMAKER_DEBUG: kedro_sagemaker_debug,
        },
    ):
        result = runner.invoke(
            cli.entrypoint, ["this", "does", "not", "matter"], catch_exceptions=False
        )
        assert result.exit_code == 0

        assert all(
            c in subprocess_run.call_args.args[0] for c in (cmd, "--params", "'{}'")
        )


@patch("subprocess.run", return_value=Mock(returncode=0))
def test_sagemaker_entrypoint_can_be_called_with_flat_params(
    subprocess_run, tmp_path: Path
):
    runner = CliRunner()
    cmd = "echo 'Unit tests'"
    with patch.dict(
        os.environ,
        {
            KEDRO_SAGEMAKER_ARGS: cmd,
            KEDRO_SAGEMAKER_WORKING_DIRECTORY: str(tmp_path.absolute()),
            KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "my.param.one",
            KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "0": (param_value := uuid4().hex),
        },
    ):
        expected_params = json.dumps({"my": {"param": {"one": param_value}}})
        result = runner.invoke(cli.entrypoint, [], catch_exceptions=False)
        assert result.exit_code == 0

        assert all(
            c in subprocess_run.call_args.args[0]
            for c in (cmd, "--params", f"'{expected_params}'")
        )
