"""Optional all-object VOC AP50; keep standard difficult-ignore metric too."""
import numpy as np
from mmdet.evaluation.metrics import VOCMetric
from mmyolo.registry import METRICS


@METRICS.register_module()
class VOCIncludingDifficultMetric(VOCMetric):
    """Treat difficult objects as ordinary GT without modifying source XML.

    This explicitly defined evaluator must not be asserted identical to the
    paper's mAP_difficult until the author's exact implementation is verified.
    The standard metric remains the pinned installed VOCMetric unchanged.
    """
    def process(self, data_batch, data_samples):
        previous = len(self.results)
        super().process(data_batch, data_samples)
        for annotation, _ in self.results[previous:]:
            required = {'bboxes', 'labels', 'bboxes_ignore', 'labels_ignore'}
            if not required.issubset(annotation):
                raise RuntimeError('Installed VOCMetric annotation contract differs')
            annotation['bboxes'] = np.concatenate(
                (annotation['bboxes'], annotation['bboxes_ignore']), axis=0)
            annotation['labels'] = np.concatenate(
                (annotation['labels'], annotation['labels_ignore']), axis=0)
            annotation['bboxes_ignore'] = annotation['bboxes_ignore'][:0]
            annotation['labels_ignore'] = annotation['labels_ignore'][:0]
