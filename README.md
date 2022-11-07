# deepsysid

A system identification toolkit for multistep prediction using deep learning and hybrid methods.

The toolkit is easy to use. 
After you follow the instructions below, you can download a dataset, run hyperparameter optimization and 
identify your best-performing models in three lines:
```shell
deepsysid download 4dof-sim-ship
deepsysid session --enable-cuda progress.json NEW
deepsysid session --enable-cuda --reportin=progress.json progress.json TEST_BEST
```

## How to use?

### Installation

Install via pip as follows, if you wish to run models on your GPU:
```shell
pip install deepsysid@git+https://github.com/AlexandraBaier/deepsysid.git
```
This command will not install the required PyTorch dependency. 
You have to install PyTorch manually.
You can find corresponding instructions [here](https://pytorch.org/get-started/locally/).

If you are fine with running everything on your CPU, you can instead use the following
command to install the package with PyTorch included:
```shell
pip install deepsysid[torch-cpu]@git+https://github.com/AlexandraBaier/deepsysid.git
```

### Environment Variables and Directories

Set the following environment variables:
```
DATASET_DIRECTORY=<root directory of dataset with at least child directory processed>
MODELS_DIRECTORY=<directory for saving of trained models>
RESULT_DIRECTORY=<directory for validation/test results>
CONFIGURATION=<JSON configuration file>
```
The `DATASET_DIRECTORY` requires a specific structure which is shown in [Datasets](#datasets).

The structure of `RESULT_DIRECTORY` and `MODELS_DIRECTORY` is automatically generated by the provided scripts.
You only need to make sure the respective directories exist.

`CONFIGURATION` points to a JSON file, which we discuss in [Configuration](#configuration).

### Datasets

The following structure is expected for the dataset directory:
```
DATASET_DIRECTORY
- processed/
-- train/
--- *.csv
-- validation/
--- *.csv
-- test/
--- *.csv
```
A ship motion dataset with the required structure can be found on 
[DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2905).

You can also download datasets in the right format with
```
deepsysid download <dataset name>
```
The dataset will be downloaded and prepared in the directory specified by `DATASET_DIRECTORY`.
We are working on continuously adding new datasets. Run ```deepsysid download``` to see what datasets are available.

### Configuration

We use a JSON file for our experiment and model configuration management. 
The configuration at its core defines a gridsearch for various models.
Our configuration are defined as a pydantic model called ```deepsysid.pipeline.configuration.ExperimentGridSearchTemplate```.
The configuration file should be placed under the path specified by `CONFIGURATION`.
We can validate the configuration file with ```deepsysid validate_configuration```.
A JSON configuration will have the following format:
```json
{
  "settings": {
    "train_fraction": "float < 1.0, these fractions are currently not used by our code.",
    "validation_fraction": "float < 1.0",
    "time_delta": "float, sampling time of your measurements",
    "window_size": "int, size of the initial window during evaluation",
    "horizon_size": "int, size of the prediction horizon during evaluation",
    "control_names": "list of strings, inputs to the model",
    "state_names": "list of strings, outputs of the model",
    "thresholds": "list of floats, only used by bounded residual models during evaluation",
    "target_metric": "name of metric used to select best performing model during grid-search",
    "metrics": {
      "name of metric": {
        "metric_class": "str, metrics (MSE, MAE, ...) executed during the evaluation.",
        "parameters": {
          "parameter name": "parameter value, metrics might additional require settings."
        }
      }
    },
    "additional_tests": {
      "name of test": {
        "test_class": "str, tests performed when calling test in addition to inference on dataset.",
        "parameters": {
          "parameter name": "parameter value, some tests require additional settings."
        }
      }
    }
  }, 
  "models": [
    {
      "model_base_name": "str, base name of model will be extended by choice of flexible hyperparameters",
      "model_class": "str",
      "static_parameters": {
        "hyperparameter name": "hyperparameter value, no grid search will be performed over these parameters"
      },
      "flexible_parameters": {
        "hyperparameter name": "list of hyperparameter values, gridsearch is performed over these parameters"
      }
    }
  ]
}
```

First, we can run `deepsysid session` to perform an automatic hyperparameter search as shown at the top of the README.

Second, we can use it to manually train/test/evaluate specific models. 
To get a list of all model names use `deepsysid write_model_names <output text>`. 
You can then, for example, run `deepsysid train <model_name>` on a chosen model.


### Command Line Interface

The `deepsysid` package exposes a command-line interface. 
Run `deepsysid` or `deepsysid --help` to access the list of available subcommands:
```
usage: Command line interface for the deepsysid package. [-h] {validate_configuration,train,test,explain,evaluate,write_model_names,session,download} ...

positional arguments:
  {validate_configuration,train,test,explain,evaluate,write_model_names,session,download}
    validate_configuration
                        Validate configuration file defined in CONFIGURATION.
    train               Train a model.
    test                Test a model.
    explain             Explain a model.
    evaluate            Evaluate a model.
    write_model_names   Write all model names from the configuration to a text file.
    session             Run a full experiment given the configuration JSON. State of the session can be loaded from and is saved to disk. This allows stopping and continuing a session at any point.
    download            Download and prepare datasets.

optional arguments:
  -h, --help            show this help message and exit
```

Run ```deepsysid {subcommand} --help``` to get details on the specific subcommand.

Some common arguments for subcommands are listed here:
- `model`: Positional argument. Name of model as defined in configuration JSON.
- `--enable-cuda`: Optional flag. Script will pass device-name `cuda` to model. Models that support GPU usage will run accordingly.
- `--device-idx={n}`: Optional argument (only in combination with `--enable-cuda`). Script will pass device name `cuda:n` to model, where `n` is an integer greater or equal to 0.
- `--disable-stdout`: Optional flag for subcommand `train`. Script will not print logging to stdout, however will still log to `training.log` in the model directory.
- `--mode={train,validation,test}`: Required argument for `test` and `evaluate`. Choose either `train`, `validation` or `test` to select what dataset to run the test and evaluate on. 

## Adding New Models

Adding new models is relatively easy. It consists of two steps (+ one optional step).

1. You need to create a configuration class by subclassing ```deepsysid.models.base.DynamicIdentificationModelConfig```,
  where you specify your model's hyperparameters.
  This is a subclass of `pydantic.BaseModel` and ensures that your configuration will always have the right types, when
  loaded from a file.
2. You need to create a model class by subclassing ```deepsysid.models.base.DynamicIdentificationModel```.
  `DynamicIdentificationModel` is an abstract class (or rather an interface in most other languages) that
  provides method signatures for initializing `__init__`, training `train`, and predicting `simulate`,
  as well as IO tasks `save`/`load`/`get_extensions` and retrieving the model size `get_parameter_count`.
  You need to implement these methods according to their signatures. Additionally, make sure to point the class
  field `CONFIG` inherited from `DynamicIdentificationModel` to your newly defined configuration class. 
  This will allow the surrounding training and testing functionalities to correctly initialize your model.
3. (Optional) Finally, you might want to ensure that your code won't break the moment it is run. 
  Adding a simple smoke test
  to ```tests/smoke_tests/test_pipeline``` is the easiest way to do so.


## References

- [Baier, A., Boukhers, Z., & Staab, S. (2021). Hybrid Physics and Deep Learning Model for Interpretable Vehicle State Prediction. ArXiv, abs/2103.06727.](https://arxiv.org/abs/2103.06727)
- [Baier, Alexandra; Staab, Steffen, 2022, "A Simulated 4-DOF Ship Motion Dataset for System Identification under Environmental Disturbances", DaRUS, 10.18419/darus-2905.](https://doi.org/10.18419/darus-2905)
