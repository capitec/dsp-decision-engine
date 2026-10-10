import typing as t
from pydantic import Field
from .s3 import S3DataSourceConfig
from .local import LocalDataSourceConfig

TDataSources = t.Annotated[
    t.Union[S3DataSourceConfig, LocalDataSourceConfig],
    Field(discriminator="provider"),
]
