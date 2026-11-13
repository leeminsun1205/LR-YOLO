"""Optimize exactly the adapter parameters; no original YOLO batch scaling."""
from mmengine.model import is_model_wrapper
from mmyolo.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class AdapterOptimizerConstructor:
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        if paramwise_cfg:
            raise ValueError('This constructor does not support paramwise_cfg')
        self.wrapper = dict(optim_wrapper_cfg)
        self.optimizer = self.wrapper.pop('optimizer')

    def __call__(self, model):
        if is_model_wrapper(model):
            model = model.module
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith('adapters.') for n in names):
            raise RuntimeError(f'Only adapter parameters may learn: {names}')
        optimizer = OPTIMIZERS.build(dict(
            self.optimizer, params=[p for p in model.parameters() if p.requires_grad]))
        return OPTIM_WRAPPERS.build(dict(self.wrapper, optimizer=optimizer))
