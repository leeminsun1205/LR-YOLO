"""VOC low-light/haze protocols. Metadata discovery only until explicit prepare."""
from pathlib import Path
import bdd100k_workflow as shared
import voc_adapter_workflow as voc
import original_voc_workflow as original

HAZE_CLASSES = ('person','car','bus','bicycle','motorbike')
HAZE_RELEASES = {
    'train':dict(count=8111,file_id='1c8DCfQKRXDcoRTwWti_seOAtHI84Q_nz',
                 archive_names=('VOC_Haze.zip','VOC_Haze_Train.zip','VOC_Fog.zip'),
                 hints=('voc_haze','voc_fog')),
    'voc':dict(count=2734,file_id='1mokbu4VuA_weakoOeMxOBd4SYsBtoaYz',
               archive_names=('VOC_Haze_Val.zip','VOC_Fog_Val.zip'),
               hints=('voc_haze_val','voc_fog_val','voc_fog_test','voc_haze_test')),
    'exdark':dict(count=4322,file_id='1cSHq3axgTdViZZ8hAwAQEqD5Jd-cMmJC',
                  archive_names=('RTTS.zip','rtts.zip'),hints=('rtts',))}


def classes_for(degradation):
    if degradation not in ('low_light','haze'):
        raise ValueError('DEGRADATION must be low_light or haze')
    return voc.VOC_CLASSES if degradation == 'low_light' else HAZE_CLASSES


def prepare_training(overrides,cache,degradation='low_light',auto_download=True,
                     archives=None,search_roots=None):
    classes_for(degradation)
    if degradation == 'low_light':
        result=original.prepare_data(overrides,cache,auto_download,archives,search_roots)
    else:
        settings=voc.prepare_datasets(dict(train=overrides.get('train',{})),Path(cache)/'haze',
            3,'train',auto_download=auto_download,archives=archives,search_roots=search_roots,
            domains=('train',),release_specs=HAZE_RELEASES)['train']
        val=overrides.get('clean_val',{})
        if val.get('root') is None:
            folder=Path(settings['root'])/settings['prefix']
            val=voc.discover_dataset(folder.parent,'voc',release_specs=HAZE_RELEASES)
            if val is None:
                raise FileNotFoundError('Set CLEAN_VAL_ROOT/MANIFEST/PREFIX to CLEAN haze validation '
                                        'in the author training bundle; never substitute fixed hazy val')
        # Fixed test packages must not be hazed a second time.
        parts=[p.lower().replace('-','_') for p in (Path(val['root'])/val['prefix']).parts]
        if any(p in ('voc_fog_test','voc_haze_test','voc_haze_val','voc_fog_val') for p in parts):
            raise ValueError('Training requires CLEAN haze validation, not offline hazy test data')
        train=voc.dataset_spec(domain='train',class_names=HAZE_CLASSES,**settings)
        validation=voc.dataset_spec(domain='voc',class_names=HAZE_CLASSES,**val)
        if train['count'] != 8111 or validation['count'] != 2734:
            raise ValueError('Expected original haze 8111 train / 2734 clean val; do not resplit')
        voc.assert_no_train_val_overlap(train,validation)
        result=dict(train=train,clean_val=validation,
                    protocol='CLEAN validation; source online haze during dark-stage validation')
    result['degradation']=degradation
    return result


def prepare_final(overrides,cache,domain,degradation='low_light',auto_download=True):
    classes_for(degradation)
    expected=('voc','exdark') if degradation=='low_light' else ('voc','rtts')
    if domain not in expected:
        raise ValueError(f'Final domains for {degradation}: {expected}')
    alias='exdark' if domain=='rtts' else domain
    settings=voc.prepare_datasets({alias:overrides.get(domain,{})},
        Path(cache)/'haze' if degradation=='haze' else cache,0,'evaluate',
        final_evaluation=alias=='exdark',domains=(alias,),auto_download=auto_download,
        release_specs=HAZE_RELEASES if degradation=='haze' else None)[alias]
    spec=voc.dataset_spec(domain=alias,**settings,
        **(dict(class_names=HAZE_CLASSES) if degradation=='haze' else {}))
    expected_count=HAZE_RELEASES[alias]['count'] if degradation=='haze' else voc.DATA_RELEASES[alias]['count']
    if spec['count'] != expected_count:
        raise ValueError('Unexpected final manifest count')
    spec['domain']=domain
    spec['degradation']=degradation
    return spec
