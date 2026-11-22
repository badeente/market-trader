"""AP7's fixed policy. Pure stdlib; deliberately not a tuning API."""
from copy import deepcopy

_POLICY = {
    'kind': 'cnn', 'architecture': 'tiny_cnn_v1', 'input_shape': [3, 24, 32],
    'conv_channels': 4, 'kernel_size': 3, 'padding': 0, 'pool_size': 2,
    'flatten_size': 660, 'hidden_units': 8, 'parameter_count': 5409,
    'activation': 'relu', 'pool': 'avg', 'output_units': 1,
    'optimizer': 'AdamW', 'learning_rate': '0.001', 'weight_decay': '0.0001',
    'betas': ['0.9', '0.999'], 'eps': '0.00000001', 'amsgrad': False,
    'foreach': False, 'fused': False, 'maximize': False,
    'batch_size': 128, 'max_epochs': 20, 'patience': 5, 'min_delta': '0.000001',
    'device': 'cpu', 'threads': 1, 'dtype': 'float64', 'deterministic': True,
    'target_scaling': 'train-only-standard', 'input_normalization': 'sum/64/255',
}


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_exact(actual[k], v) for k,v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_exact(a,b) for a,b in zip(actual,expected))
    return actual == expected


def canonical_cnn_settings(value: dict | None = None, *, snapshot=False) -> dict:
    """Kind-only user settings, or a complete exact frozen snapshot."""
    if snapshot:
        if not _exact(value, _POLICY):
            raise ValueError('CNN snapshot must match the complete frozen AP7 policy and types')
    elif value is not None and not _exact(value, {'kind':'cnn'}):
        raise ValueError('CNN accepts kind only; tuning fields are unsupported')
    return deepcopy(_POLICY)
