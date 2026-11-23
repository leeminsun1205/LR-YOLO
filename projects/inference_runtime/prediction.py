"""Optional decoder bypass; checkpoints and native model classes stay unchanged."""
from contextlib import contextmanager
from types import MethodType


def encoder_features(backbone, inputs):
    """The native reconstruct backbones run this exact encoder loop before decoding."""
    if not hasattr(backbone, 'layers') or not hasattr(backbone, 'out_indices'):
        raise TypeError('Expected the native LR-YOLO reconstruction backbone')
    outputs = []
    for index, name in enumerate(backbone.layers):
        inputs = getattr(backbone, name)(inputs)
        if index in backbone.out_indices:
            outputs.append(inputs)
    if len(outputs) != 3:
        raise ValueError('Expected three native detection feature levels')
    return tuple(outputs)


def detection_features(model, inputs, decoder_free):
    if not decoder_free:
        features, _ = model.extract_feat(inputs)
        return features
    features = encoder_features(model.backbone, inputs)
    return model.neck(features) if model.with_neck else features


@contextmanager
def prediction_mode(model, *, decoder_free=True, render_reconstruction=False):
    """Temporarily change prediction only. Full mode can omit RGB rendering too.

    Using render_reconstruction=False in BOTH benchmark modes isolates decoder
    computation, rather than attributing image interpolation/export to a speedup.
    No LPIPS/decoder weights are removed and strict checkpoint loading is retained.
    """
    if model.training:
        raise RuntimeError('Inference mode requires model.eval(); training is untouched')
    if decoder_free and render_reconstruction:
        raise ValueError('RGB reconstruction requires the decoder')
    if render_reconstruction:
        yield model
        return
    had_local = 'predict' in model.__dict__
    previous = model.__dict__.get('predict')

    def predict(this, batch_inputs, batch_data_samples, rescale=True):
        features = detection_features(this, batch_inputs, decoder_free)
        predictions = this.bbox_head.predict(features, batch_data_samples, rescale=rescale)
        return this.add_pred_to_datasample(batch_data_samples, predictions)

    model.predict = MethodType(predict, model)
    try:
        yield model
    finally:
        if had_local:
            model.predict = previous
        else:
            delattr(model, 'predict')
