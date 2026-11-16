"""Validated registries and VOC gallery metadata; no Torch import."""
import hashlib
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = {
    'low_light': ('person','car','bus','bicycle','motorbike','boat','bottle','chair','dog','cat'),
    'haze': ('person','car','bus','bicycle','motorbike'),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_registry(path):
    entries=json.loads(Path(path).read_text())
    if not isinstance(entries,list): raise ValueError('Model registry must be a list')
    result={}
    for raw in entries:
        e=dict(raw); key=e['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',key) or key in result:
            raise ValueError('Invalid/duplicate model ID')
        if e['degradation'] not in CLASSES or e['architecture'] not in ('v8','v10'):
            raise ValueError('Invalid model family/architecture')
        if e.get('stage') not in ('clean','dark') or e.get('case_id') not in range(7):
            raise ValueError('Model stage/case must be explicit')
        if e['stage']=='clean' and e['case_id']!=0: raise ValueError('Clean requires S0')
        e['classes']=list(CLASSES[e['degradation']])
        e['available']=all(Path(e.get(k,'')).is_file() for k in ('checkpoint','config'))
        result[key]=e
    return result


class Gallery:
    def __init__(self, specs):
        self.items={}
        for spec in specs:
            family=spec['degradation']
            if family not in CLASSES: raise ValueError('Invalid gallery family')
            root=Path(spec['root']).resolve()
            folder=(root/spec.get('prefix','')).resolve()
            if not folder.is_relative_to(root): raise ValueError('Gallery prefix escapes root')
            manifest=Path(spec['manifest'])
            if not manifest.is_absolute(): manifest=root/manifest
            ids=manifest.read_text().split()
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate manifest IDs')
            for ident in ids:
                if not re.fullmatch(r'[A-Za-z0-9_-]+',ident): raise ValueError('Invalid VOC ID')
                image=folder/'JPEGImages'/f'{ident}.jpg'
                xml=folder/'Annotations'/f'{ident}.xml'
                if not image.is_file(): raise FileNotFoundError(image)
                key=f'{family}_{spec["id"]}_{ident}'
                if key in self.items: raise ValueError('Duplicate gallery key')
                self.items[key]=dict(id=key,image_id=ident,path=image,xml=xml,
                    degradation=family,source=spec.get('source','VOC'),split=spec['split'],
                    already_degraded=bool(spec.get('already_degraded',False)))

    def public(self,item):
        return {k:v for k,v in item.items() if k not in ('path','xml')}

    def choose(self,family):
        pool=[v for v in self.items.values() if v['degradation']==family]
        if not pool: raise ValueError('Chưa có gallery cho degradation này')
        return self.public(random.SystemRandom().choice(pool))

    def annotations(self,key):
        item=self.items[key]
        if not item['xml'].is_file(): return []
        root=ET.parse(item['xml']).getroot(); size=root.find('size')
        # Raw VOC convention is declared; never report a benchmark metric here.
        result=[]
        for obj in root.findall('object'):
            label=obj.findtext('name')
            if label not in CLASSES[item['degradation']]: continue
            b=obj.find('bndbox')
            if b is None: continue
            result.append(dict(label=label,bbox=[float(b.findtext(k)) for k in
                ('xmin','ymin','xmax','ymax')],difficult=obj.findtext('difficult')=='1'))
        return result


def load_reports(path):
    if not path or not Path(path).is_file(): return []
    rows=json.loads(Path(path).read_text())
    if not isinstance(rows,list): raise ValueError('Reports must be a list')
    ids=set()
    for row in rows:
        # Curated measured reports only; no fictional defaults.
        for key in ('id','label','source','degradation','architecture','dataset','metric','epochs'):
            if key not in row: raise ValueError(f'Report missing {key}')
        if row['id'] in ids: raise ValueError('Duplicate report ID')
        ids.add(row['id'])
        if row['degradation'] not in CLASSES: raise ValueError('Report family invalid')
        if not isinstance(row['epochs'],int) or row['epochs']<0: raise ValueError('Report epochs invalid')
        for key in ('ap50','train_hours','vram_gb'):
            value=row.get(key)
            if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):
                raise ValueError(f'Report {key} must be finite and nonnegative')
        if row.get('ap50') is not None and row['ap50']>1: raise ValueError('AP50 must be a fraction, not percent')
    return rows
