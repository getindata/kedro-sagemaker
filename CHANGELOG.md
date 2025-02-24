# Changelog

## [Unreleased]

## [0.4.0] - 2025-02-19

-   Ensured compatibility of kedro-sagemaker with Kedro versions > 0.19 by updating dependencies, dataset naming, configuration loader, and documentation.

## [0.3.0] - 2023-02-20

-   Add subnets and security groups configuration for nodes (<https://github.com/getindata/kedro-sagemaker/pull/11>) by [@wmikolajczyk-fandom](https://github.com/wmikolajczyk-fandom)

## [0.2.1] - 2023-02-17

-   Add links to video tutorial in docs / readme

## [0.2.0] - 2023-02-08

-   Support for Mlflow with shared run across pipeline steps
-   Fixed ability to overwrite docker image in `kedro sagemaker run`

## [0.1.1] - 2022-12-30

-   Pass missing environment to the internal entrypoint
-   Add lazy initialization and cache to Kedro's context in the `KedroContextManager` class to prevent re-loading

## [0.1.0] - 2022-12-27

-   Initial release with basic feature set

## [0.0.1] - 2022-11-24

-   Project seed prepared

[Unreleased]: https://github.com/getindata/kedro-sagemaker/compare/0.4.0...HEAD

[0.4.0]: https://github.com/getindata/kedro-sagemaker/compare/0.3.0...0.4.0

[0.3.0]: https://github.com/getindata/kedro-sagemaker/compare/0.2.1...0.3.0

[0.2.1]: https://github.com/getindata/kedro-sagemaker/compare/0.2.0...0.2.1

[0.2.0]: https://github.com/getindata/kedro-sagemaker/compare/0.1.1...0.2.0

[0.1.1]: https://github.com/getindata/kedro-sagemaker/compare/0.1.0...0.1.1

[0.1.0]: https://github.com/getindata/kedro-sagemaker/compare/0.0.1...0.1.0

[0.0.1]: https://github.com/getindata/kedro-sagemaker/compare/1de2c65027d02a885dda61727dfe83a3404badcf...0.0.1
