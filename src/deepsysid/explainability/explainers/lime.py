import warnings

from .blackbox import LIMEExplainer, LIMEExplainerConfig

warnings.warn(
    'Import LIMEExplainer and LIMEExplainerConfig from deepsysid.'
    'explainers.blackbox in the future. The module deepsysid.explainers.'
    'lime will be deprecated in a future release.'
)

__all__ = ['LIMEExplainerConfig', 'LIMEExplainer']
