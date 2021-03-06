# deepsysid

A system identification toolkit for multistep prediction using deep learning and hybrid methods.

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

### Environment Variables

Set the following environment variables:
```
DATASET_DIRECTORY=<root directory of dataset with at least child directory processed>
RESULT_DIRECTORY=<directory for validation/test results>
CONFIGURATION=<JSON configuration file>
```

### File Structure

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

The structure of the `RESULT_DIRECTORY` is automatically generated by the provided scripts.
You only need to make sure the directory, which `RESULT_DIRECTORY` points at exists.

### Configuration

The JSON configuration file (pointed at by environment variable `CONFIGURATION`) is used 
to set hyperparameters for any model in the `deepsysid` package.
The JSON file has the following format:
```json
{
  "train_fraction": "float < 1.0",
  "validation_fraction": "float < 1.0",
  "time_delta": "float, sampling rate of dataset in seconds",
  "window_size": "int, size of window of initial state and control inputs during evaluation",
  "horizon_size": "int, prediction horizon used during evaluation",
  "control_names": "list of strings, names of control/external inputs as defined in the dataset CSV",
  "state_names": "list of strings, names of measured (and also predicted) system states as defined in the dataset CSV",
  "thresholds": "optional, list of floats, threshold values for which the hybrid models should be evaluated",
  "models": {
    "str, unique name of model": {
      "model_class": "str, full Python class name of model including full package, for example, deepsysid.models.linear.LinearModel",
      "location": "str, directory where model files should be saved such as weights and normalization parameters",
      "parameters": {
        "hyperparameter_name": "hyperparameter_value, an example follows",
        "dropout": 0.25
      }
    }
  }
}
```

Alternatively, a gridsearch template JSON format can be used to generate the above configuration file
using the script `scripts/build_gridsearch_configuration.py`.
It allows the generation of models with a large number of hyperparameter combinations. 
The gridsearch template has the following format:
```json
{
  "base_path": "str, directory where all trained models should be saved to, subdirectories will be created for each separate model",
  "settings": {
    "train_fraction": "float < 1.0",
    "validation_fraction": "float < 1.0",
    "time_delta": "float",
    "window_size": "int",
    "horizon_size": "int",
    "control_names": "list of strings",
    "state_names": "list of strings",
    "thresholds": "list of floats",
    "models": [
      {
        "model_base_name": "str, base name of model will be extended by combination of hyperparameter",
        "model_class": "str",
        "static_parameters": {
          "hyperparameter name": "hyperparameter value, static parameters remain the same for all models of this base_name, no grid search will be performed over these parameters"
        },
        "flexible_parameters": {
          "hyperparameter name": "list of hyperparameter values, gridsearch is performed over flexible parameters, the cartesian product over all possible flexible parameter combinations is generated as distinct models"
        }
      }
    ]
  }
}
```
If you use the template format, the environment variable should still point to the JSON configuration file, that is generated by the script.
An example template can be found under `examples/patrol_ship.template.json`.

### Command Line Interface

The `deepsysid` package exposes a command-line interface. 
Run `deepsysid` or `deepsysid --help` to access the list of available subcommands:
```
usage: Command line interface for the deepsysid package. [-h]
                                                         {build_configuration,train,test,evaluate,write_model_names}
                                                         ...

positional arguments:
  {build_configuration,train,test,evaluate,write_model_names}
    build_configuration
                        Build configuration file given grid-search configuration template. Resulting
                        configuration is written to CONFIGURATION.
    train               Train a model.
    test                Test a model.
    evaluate            Evaluate a model.
    write_model_names   Write all model names from the configuration to a text file.

optional arguments:
  -h, --help            show this help message and exit
```

Run `deepsysid {subcommand} --help` to get details on the specific subcommand.

Some common arguments for subcommands are listed here:
- `model`: Positional argument. Name of model as defined in configuration JSON.
- `--enable-cuda`: Optional flag. Script will pass device-name `cuda` to model. Models that support GPU usage will run accordingly.
- `--device-idx={n}`: Optional argument (only in combination with `--enable-cuda`). Script will pass device name `cuda:n` to model, where `n` is an integer greater or equal to 0.
- `--disable-stdout`: Optional flag for subcommand `train`. Script will not print logging to stdout, however will still log to `training.log` in the model directory.
- `--mode={train,validation,test}`: Required argument for `test` and `evaluate`. Choose either `train`, `validation` or `test` to select what dataset to run the test and evaluation on. 

## References

[Baier, A., Boukhers, Z., & Staab, S. (2021). Hybrid Physics and Deep Learning Model for Interpretable Vehicle State Prediction. ArXiv, abs/2103.06727.](https://arxiv.org/abs/2103.06727)
