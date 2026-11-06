"""Read this project's zero-based BDD XML without a VOC one-based offset.

The converter writes rounded zero-based xyxy coordinates. MMDetection VOC
parsers may subtract one. Keep their image metadata/ignore flags, but restore
the explicitly stored BDD coordinates in memory. No files are rewritten.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from mmyolo.datasets.yolov5_voc_dark import YOLOv5BDD100KNightDataset
from mmyolo.registry import DATASETS


def read_bdd_boxes(xml_path, classes):
    """Return XML objects in parser order, with zero-based coordinates."""
    labels = {name: index for index, name in enumerate(classes)}
    objects = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        name = obj.findtext('name')
        if name not in labels:
            raise ValueError(f'Unexpected BDD class {name!r} in {xml_path}')
        box = obj.find('bndbox')
        if box is None:
            raise ValueError(f'Missing box in {xml_path}')
        coords = [float(box.findtext(key))
                  for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not (0 <= coords[0] < coords[2]
                and 0 <= coords[1] < coords[3]):
            raise ValueError(f'Invalid zero-based BDD box {coords}')
        objects.append((labels[name], coords))
    return objects


@DATASETS.register_module()
class YOLOv5BDD100KDayNightDataset(YOLOv5BDD100KNightDataset):
    """Same class order and VOC pipelines, explicit BDD coordinate convention."""

    def parse_data_info(self, raw_data_info):
        info = super().parse_data_info(raw_data_info)
        # This project uses <split>/JPEGImages and <split>/Annotations.
        image = Path(info['img_path'])
        xml_path = image.parent.parent / 'Annotations' / (image.stem + '.xml')
        objects = read_bdd_boxes(xml_path, self.metainfo['classes'])
        instances = info['instances']
        if len(objects) != len(instances):
            raise ValueError(f'XML/parser object count mismatch: {xml_path}')
        for instance, (label, coords) in zip(instances, objects):
            if instance['bbox_label'] != label:
                raise ValueError(f'XML/parser object order mismatch: {xml_path}')
            instance['bbox'] = coords
        return info
