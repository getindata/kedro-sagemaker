import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from kedro_sagemaker import cli
from kedro_sagemaker.config import KedroSageMakerPluginConfig
from kedro_sagemaker.utils import CliContext


@pytest.fixture()
def patched_kedro_package():
    with patch("kedro.framework.project.PACKAGE_NAME", "tests") as patched_package:
        original_dir = os.getcwd()
        os.chdir("tests")
        yield patched_package
        os.chdir(original_dir)


@pytest.fixture()
def cli_context() -> CliContext:
    metadata = MagicMock()
    metadata.package_name = "tests"
    return CliContext("base", metadata)


def test_can_initialize_basic_plugin_confiig(
    patched_kedro_package,
    cli_context,
):
    # given
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("conf/base").mkdir(parents=True)

        # when
        result = runner.invoke(
            cli.init,
            [
                "__bucket_name__",
                "__execution_role__",
                "__docker_image__",
                "--yes",
            ],
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
