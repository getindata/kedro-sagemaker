# Taken from https://github.com/kedro-org/kedro-plugins/tree/main/kedro-docker
# and modified on 2022.12.13

DOCKERFILE_TEMPLATE = """
ARG BASE_IMAGE=python:3.9.15
FROM $BASE_IMAGE

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && rm -f /tmp/requirements.txt

WORKDIR /home/kedro
COPY . .

# Do not change the default entrypoint, it will break the Kedro SageMaker integration!
ENTRYPOINT ["kedro", "sagemaker", "entrypoint"]
""".strip()

DOCKERIGNORE_TEMPLATE = """
##########################
# Kedro PROJECT

# ignore Dockerfile and .dockerignore
Dockerfile
.dockerignore

# ignore potentially sensitive credentials files
conf/**/*credentials*

# ignore all local configuration
conf/local
!conf/local/.gitkeep

# ignore everything in the following folders
data
logs
notebooks
references
results

# except the following
!logs/.gitkeep
!notebooks/.gitkeep
!references/.gitkeep
!results/.gitkeep
!data/01_raw
""".strip()
