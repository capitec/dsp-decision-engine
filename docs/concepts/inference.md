# API Inference Handling

In SpockFlow, method overloading plays a pivotal role in customizing the behavior of the inference handler, allowing users to tailor every aspect of API deployment to their specific needs. Several key methods and classes within the framework can be easily overridden to adjust functionality:

- **Input Handling**: Customize data decoding and preprocessing using `input_fn`, `decoders`.
- **Model Management**: Define how models are loaded and configured with `model_fn`, `dag_loader_fn`, `config_manager_loader_fn`, `model_loader_cls`, `model_cache_cls`, and `model_config_cls`.
- **Prediction and Post-processing**: Modify prediction logic and result refinement via `predict_fn` and `post_process_fn`.
- **Response Formatting**: Adjust how predictions are encoded into various output formats using `output_fn`, `encoders`.

These methods not only provide flexibility but also enhance integration capabilities, optimize performance, and align seamlessly with existing systems. This section will delve into each function's role in tailoring the deployment of SpockFlow's API, offering insights into practical customization strategies and implementation guidelines.

## Overview

The `transform_fn` serves as the central entry point in SpockFlow's API deployment, where raw requests are processed and transformed into meaningful responses. This function orchestrates a sequence of operations, starting with data decoding and preprocessing, followed by model prediction and output formatting. Each step in this workflow can be tailored to specific requirements by defining corresponding functions in the `inference.py` module. The following diagram is a high-level overview of the `transform_fn`.


```{figure} ../_static/concepts/inference.drawio.svg
:scale: 100
:align: center
:class: only-dark
```

```{figure} ../_static/concepts/inference_light.drawio.svg
:scale: 100
:align: center
:class: only-light
```

For instance, the input_fn method can be customized within `inference.py` to handle data decoding based on content type. Here’s an example:
```python
# inference.py

def input_fn(input_data, content_type):
    # Custom decoding logic
    decoded_data = custom_decoder(input_data, content_type)
    return decoded_data
```
Additionally, methods defined with `self` as the first argument in `inference.py` allow the instance of the `ServingHandler` to be injected dynamically at runtime. This enables seamless integration of custom functionalities and ensures flexibility in adapting the inference pipeline to diverse application scenarios.

The following sections will explore each method's role within `transform_fn`, illustrating how they can be leveraged to tailor the API deployment process in SpockFlow.

## Methods

This section describes all the configurable stages that can be overloaded in the inference.py and how they affect the predicted result.
### Input Function
The `input_fn` method in SpockFlow's ServingHandler class handles the initial step of data ingestion and decoding during API requests. It accepts two parameters:
- **input_data**: The raw data received from the API request, typically in bytes.
- **content_type**: Specifies the format or encoding of input_data, guiding the decoding process.
The primary role of `input_fn` is to decode and pre-process incoming raw data into a structured format that downstream processes can handle effectively. This method plays a crucial role in ensuring compatibility between various data sources and the internal processing pipeline of SpockFlow.

**Example:**

```python
# inference.py
import json

def input_fn(input_data, content_type):
    if content_type != 'application/json':
        raise ValueError("API only supports json data")
    return json.loads(input_data)
```

The default input_fn function in SpockFlow utilizes a decoders dictionary to decode incoming data based on the content-type header of API requests. By default, it supports decoding both CSV and JSON formats, converting them into Python objects. For instance, CSV data is parsed into a Pandas DataFrame, while JSON data is directly parsed into a dictionary. Users have the flexibility to extend or override these default decoders in the inference.py module. For example, adding support for Parquet format can be achieved by defining a custom decoder function and updating the decoders dictionary accordingly:

```python
# inference.py
import pandas as pd
from io import BytesIO
from spockflow.inference.io.decoders import default_decoders

decoders = {**default_decoders}

def decode_parquet(data: bytes):
    return pd.read_parquet(BytesIO(data)).to_dict(orient='records')

decoders["application/parquet"] = decode_parquet
```

### Preprocessing Function
The pre_process_fn function in SpockFlow facilitates the transformation of JSON data into a format suitable for downstream processing, typically a Pandas DataFrame. Its primary role is to prepare input data for enrichment, scoring, or other operations supported by SpockFlow. A typical implementation is as follows:
```python
# inference.py
import pandas as pd

def pre_process_fn(input_data: dict) -> pd.DataFrame:
    return pd.json_normalize(input_data)
```

### Output Function

The `output_fn` function in SpockFlow's API deployment handles the final step of encoding prediction results into various output formats based on the `accept` header of API responses. It accepts two parameters:

- **prediction**: The dictionary containing the processed prediction results.
- **accept**: Specifies the desired format or encoding of the response, guiding the encoding process.

The primary role of `output_fn` is to format prediction results into the specified output format, ensuring compatibility with client expectations. This function is crucial for delivering responses in formats such as JSON, CSV, or custom formats like Parquet.

**Example Implementation:**

```python
import pandas as pd
from starlette.responses import JSONResponse

def output_fn(prediction: dict, accept: str) -> "JSONResponse":
    """
    Encodes prediction results into JSON format based on the accept header.

    Parameters:
    - prediction (dict): Processed prediction results to be encoded.
    - accept (str): Desired format or encoding specified by the accept header.

    Returns:
    - JSONResponse: Response object containing encoded prediction results.
    """
    if accept not in ["*/*", "application/json"]:
        raise ValueError("Invalid accept header")

    return JSONResponse(prediction)
```


#### Post-process Function

In conjunction with `output_fn`, the `post_process_fn` function is commonly used to manipulate prediction results before they are encoded. It adjusts the structure or content of the output data to meet specific client needs.

**Example Implementation:**

```python
def post_process_fn(result: dict) -> dict:
    """
    Post-processes prediction result before encoding.

    Parameters:
    - result (dict): Raw prediction result from the model.

    Returns:
    - dict: Processed prediction result ready for encoding.
    """
    return {"response": result["tree"]}
```

The `post_process_fn` function in SpockFlow allows developers to refine prediction outputs, ensuring they meet application-specific requirements before encoding them into the desired format using `output_fn`.

#### Encoders Example

Developers can extend the functionality of SpockFlow's default `output_fn` by defining custom encoders for additional output formats. The `output_fn` function utilizes the encoders dictionary to determine the output format based on the accept header in API responses. Below is an example of adding support for Parquet format using a custom encoder:

```python
import pandas as pd
from starlette.responses import Response
from spockflow.inference.io.encoders import default_encoders

def to_parquet(result: dict) -> Response:
    """
    Encodes prediction result into Parquet format.

    Parameters:
    - result (dict): Processed prediction result to be encoded.

    Returns:
    - Response: Response object containing Parquet-encoded prediction result.
    """
    res = BytesIO()
    pd.json_normalize(result).to_parquet(res)
    res.seek(0)
    return Response(res.read(), media_type="application/vnd.apache.parquet")

# Initialize encoders with default encoders
encoders = {**default_encoders}

# Add support for Parquet format
encoders["application/vnd.apache.parquet"] = to_parquet
```

## Configuration Management in SpockFlow

SpockFlow provides flexible configuration management capabilities to facilitate the setup and management of models and their associated configurations. Developers can utilize built-in managers like the `YamlConfigManager` or create custom managers tailored to specific needs, such as reading configurations from DynamoDB or other data sources.

### Using YamlConfigManager

The `YamlConfigManager` provided by SpockFlow simplifies configuration management using YAML files. It allows for straightforward handling of model configurations stored locally. Here's an example of configuring `model_config_cls` to use `YamlConfigManager`:

```python
from spockflow.inference.config.loader.yamlmanager import YamlConfigManager
model_config_cls = YamlConfigManager
```

### Custom Config Managers

Developers have the flexibility to implement custom config managers to suit unique requirements. For instance, a custom config manager could integrate with DynamoDB to dynamically fetch configurations. Below is a simplified outline of how a custom manager might be structured:

```python
# Example of a custom config manager (simplified)

class CustomConfigManager(ConfigManager):
    def get_latest_version(self, model_name: str) -> str:
        # Implement logic to fetch the latest version from DynamoDB or other sources
        pass

    def get_config(self, model_name: str, model_version: str) -> TNamespacedConfig:
        # Implement logic to fetch config from DynamoDB or other sources
        pass
    
    def save_to_config(self, model_name: str, model_version: str, namespace: str, config: TNamespacedConfig, key: str | None = None):
        # Implement logic to save config to DynamoDB or other sources
        pass
```

### YamlConfigManager Implementation

For reference, here is a simplified implementation of `YamlConfigManager` provided by SpockFlow, demonstrating its capability to manage YAML-based configurations locally:

```python
from yaml import dump, load
try:
    from yaml import CDumper as Dumper, CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader
import os
from .base import ConfigManager, TNamespacedConfig
from pydantic_settings import BaseSettings, SettingsConfigDict

class YamlConfigManager(ConfigManager, BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix='CONFIG_MANAGER_')
    config_path: str = os.path.join(".", "config")

    def model_path(self, model_name: str) -> str:
        return os.path.join(self.config_path, model_name)

    def get_latest_version(self, model_name: str) -> str:
        if model_name == "__default__":
            paths = os.listdir(self.config_path)
            if model_name not in paths:
                assert len(paths) == 1, "can only use default when there is one model or an explicit __default__"
                model_name = paths[0]
        return sorted(os.listdir(self.model_path(model_name)))[-1]

    def get_config(self, model_name: str, model_version: str) -> TNamespacedConfig:
        r = {}
        for f in glob(os.path.join(self.model_path(model_name), model_version, "*.yml")):
            ns = os.path.splitext(os.path.split(f)[1])[0]
            with open(f) as fp:
                r[ns] = load(fp, Loader=Loader)
        return r

    def save_to_config(self, model_name: str, model_version: str, namespace: str, config: TNamespacedConfig, key: str | None = None):
        save_path = os.path.join(self.model_path(model_name), model_version, namespace) + ".yml"
        if key is not None:
            if os.path.isfile(save_path):
                with open(save_path) as fp:
                    curr_config = load(fp, Loader=Loader)
                curr_config[key] = config
                config = curr_config
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        with open(save_path, "w") as fp:
            dump(config, fp, Dumper=Dumper)
```

### Conclusion

SpockFlow's configuration management capabilities, exemplified by the `YamlConfigManager` and the possibility of creating custom managers, provide robust support for handling model configurations across different use cases. Whether utilizing YAML files locally or integrating with external data sources like DynamoDB, developers have the tools necessary to effectively manage and deploy models within SpockFlow.


## Advanced Methods
While the previously discussed methods cover the majority of use cases, SpockFlow also offers advanced customization options for developers seeking more specialized functionalities. These advanced methods provide additional flexibility and control over API deployment processes, enabling tailored solutions for specific requirements.
### Predict Function
The `predict_fn` function in SpockFlow is pivotal for executing model predictions using a specified `Driver` object. By default, it leverages `raw_execute` to provide flexibility in handling prediction outputs, returning results as a dictionary. However, developers can override this behavior, as shown in the example below, to use `execute` instead. This alternative approach directly returns a DataFrame, eliminating the need for a separate `post_process_fn` for data transformation.

### Example Implementation

```python
from spockflow.inference.handler import WrappedInputData
from spockflow.core import Driver

def predict_fn(input_data: "WrappedInputData", model: "Driver") -> pd.DataFrame:
    """
    Executes model prediction using SpockFlow's Driver object with direct DataFrame output.

    Parameters:
    - input_data (WrappedInputData): Wrapped input data containing model inputs.
    - model (Driver): SpockFlow's Driver object representing the model to execute.

    Returns:
    - pd.DataFrame: DataFrame containing the prediction results.
    """
    return model.execute(
        inputs=input_data.data,
        overrides=input_data.input_overrides
    )
```

In this revised example, `predict_fn` overrides the default behavior by using `execute` instead of `raw_execute`. This method directly returns a DataFrame containing prediction results. By adopting this approach, developers streamline the prediction process within SpockFlow, ensuring efficient data processing without the need for additional transformation steps typically handled by a `post_process_fn`.
