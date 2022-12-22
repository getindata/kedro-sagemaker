
# Development

## Prerequisites
* poetry `1.3.11` (as of 2022-12-22)
* Python >= 3.9
* AWS CLI - https://aws.amazon.com/cli/

## Local development
It's easiest to develop the plugin by having a side project created with Kedro (e.g. spaceflights starter), managed by Poetry (since there is no `pip install -e` support in Poetry).
In the side project, just add the following entry in `pyproject.toml`:
```toml
[tool.poetry.dependencies]
kedro-sagemaker = { path = "<full path to the plugin on local machine>", develop = true}
```
and invoke
```console
poetry update
poetry install
```
and all the changes made in the plugin will be immediately visible in the side project (just as with `pip install -e` option).

## Starting the job from local machine
Since you need a docker container to run the job in Azure ML Pipelines, it needs to be build first. For fast local development I suggest the following:
1. Once you decide to test the plugin, run `poetry build`. It will create `dist` folder with `.tar.gz` file in it.
2. Go to the side project folder, create a hard-link to the `.tar.gz`: `ln <full path to the plugin on local machine>/dist/kedro-sagemaker-0.1.0.tar.gz kedro-sagemaker-0.1.0.tar.gz`
3. In the `Dockerfile` of the side project add
```Dockerfile
COPY kedro-sagemaker-0.1.0.tar.gz .
RUN pip install ./kedro-sagemaker-0.1.0.tar.gz
```
4. Build the docker with `:latest` tag (make sure that `:latest` is specified in the plugin's config `sagemaker.yml` in `conf`), push the image and run the plugin.
5. Done!
