import yaml

from kedro_sagemaker.config import CONFIG_TEMPLATE_YAML


def test_can_template_config():
    cfg = CONFIG_TEMPLATE_YAML.format(
        **{
            "bucket": "asd",
            "docker_image": "image:latest",
            "execution_role": "arn:aws:iam::1234:/56/Role",
        }
    )
    assert isinstance(yaml.safe_load(cfg), dict)
