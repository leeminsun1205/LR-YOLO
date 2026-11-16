"""Author VOC releases from official README. Download only on explicit call."""
import json
import os
import shutil
import stat
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from .server.core import digest

SOURCE='https://github.com/diasuki/LR-YOLO#pretrained-models'
RELEASES={
 'voc_dark_v10':dict(architecture='v10',degradation='low_light',file_id='1eIwemKii0QJmSaBByYKZINmmVX69Xtcs',
    member='best_pascal_voc_mAP_epoch_97.pth'),
 'voc_dark_v8':dict(architecture='v8',degradation='low_light',file_id='1fbliGLevVcPiv7ZY0XcWOqApavvauTRB'),
 'voc_haze_v10':dict(architecture='v10',degradation='haze',file_id='1fvkrAMd3bcuvhbdB9HL_6NIVawtEd8cX'),
 'voc_haze_v8':dict(architecture='v8',degradation='haze',file_id='1PLWj2-h3AzM8WnJ1Vn6CeDBCoDTFaNCb'),
}


def choose_member(names,override=None,preferred=None):
    candidates=[n for n in names if n.lower().endswith('.pth')]
    for n in candidates:
        p=PurePosixPath(n)
        if p.is_absolute() or '..' in p.parts or '\\' in n:raise ValueError('Unsafe checkpoint archive member')
    if override:
        if override not in candidates:raise ValueError('Requested member absent from archive: '+override)
        return override
    if preferred:
        matches=[n for n in candidates if PurePosixPath(n).name==preferred]
        if len(matches)==1:return matches[0]
        if len(matches)>1:raise ValueError('Ambiguous preferred member: '+repr(matches))
    best=[n for n in candidates if PurePosixPath(n).name.startswith('best_')]
    if len(best)==1:return best[0]
    if not best and len(candidates)==1:return candidates[0]
    raise ValueError('Set AUTHOR_CHECKPOINT_MEMBERS[release_id] to one EXACT archive member. Candidates: '+repr(candidates))


def extract_checkpoint(archive,destination,override=None,preferred=None):
    """Copy a single regular checkpoint, never unpack archive paths/config code."""
    archive,destination=Path(archive),Path(destination)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if any(n.lower().endswith('.pth') for n in names):
                name=choose_member(names,override,preferred);info=z.getinfo(name)
                if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink checkpoint rejected')
                if shutil.disk_usage(destination.parent).free<info.file_size+256*1024**2:
                    raise RuntimeError('Insufficient disk to extract author checkpoint')
                with z.open(info) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
                return name
            # A torch.save checkpoint is itself ZIP (data.pkl + tensor storage).
            if any(n.endswith('/data.pkl') or n=='data.pkl' for n in names):
                if override:raise ValueError('Raw Torch checkpoint has no selectable member')
                shutil.copy2(archive,destination);return '(raw torch ZIP checkpoint)'
            raise ValueError('Downloaded ZIP contains no checkpoint')
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members=t.getmembers();name=choose_member([m.name for m in members],override,preferred)
            member=t.getmember(name)
            if not member.isfile():raise ValueError('Non-regular checkpoint rejected')
            if shutil.disk_usage(destination.parent).free<member.size+256*1024**2:
                raise RuntimeError('Insufficient disk to extract author checkpoint')
            with t.extractfile(member) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
            return name
    with archive.open('rb') as f:header=f.read(2)
    if header[:1]!=b'\x80':raise ValueError('Download is neither a checkpoint nor ZIP/TAR (possibly a Drive error page)')
    if override:raise ValueError('Raw checkpoint has no selectable member')
    shutil.copy2(archive,destination);return '(raw pickle checkpoint)'


def prepare_author_release(repo,cache,release_id,member=None,downloader=None):
    if release_id not in RELEASES:raise ValueError('Unknown author release: '+str(release_id))
    spec=RELEASES[release_id];version=spec['architecture'];family=spec['degradation']
    native='dark' if family=='low_light' else 'fog';weather='dark' if family=='low_light' else 'haze'
    rel=f'configs/yolo{version}_voc_{native}/3_finetune_yolo{version}_s_reconstruct_voc_{native}_ep100_{weather}_train_no_aug_{weather}_test.py'
    # Source haze configs omit ep100 in their filenames.
    if family=='haze':rel=rel.replace('_ep100_','_100e_')
    config=Path(repo)/rel
    if not config.is_file():raise FileNotFoundError(config)
    folder=Path(cache)/release_id;folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'author_download.bin';checkpoint=folder/'author_checkpoint.pth';sidecar=folder/'release.json'
    if checkpoint.exists():
        if not sidecar.is_file():raise FileExistsError('Checkpoint exists without release provenance: '+str(checkpoint))
        record=json.loads(sidecar.read_text())
        if (record.get('release_id')!=release_id or record.get('file_id')!=spec['file_id']
                or (member and record.get('member')!=member)
                or checkpoint.stat().st_size!=record.get('checkpoint_size')):
            raise FileExistsError('Existing release cache differs; preserve it and select another cache folder')
        print('Author checkpoint cache:',checkpoint,flush=True)
    else:
        if not archive.is_file():
            if shutil.disk_usage(folder).free<512*1024**2:raise RuntimeError('Less than512MB disk available for release download')
            if downloader is None:
                import gdown
                downloader=gdown.download
            pending=folder/('download_'+uuid.uuid4().hex+'.part')
            print('Downloading author release:',release_id,'Drive ID:',spec['file_id'],flush=True)
            downloader(id=spec['file_id'],output=str(pending),quiet=False)
            if not pending.is_file() or not pending.stat().st_size:raise RuntimeError('Release download produced no data')
            with pending.open('rb') as f:header=f.read(2)
            if not zipfile.is_zipfile(pending) and not tarfile.is_tarfile(pending) and header[:1]!=b'\x80':
                raise ValueError('Drive returned no checkpoint/archive (possibly quota/HTML). '
                                 'Partial download preserved; rerun will retry, not reuse this error page.')
            os.replace(pending,archive)
        pending=folder/('checkpoint_'+uuid.uuid4().hex+'.part')
        selected=extract_checkpoint(archive,pending,override=member,preferred=spec.get('member'))
        record=dict(release_id=release_id,file_id=spec['file_id'],member=selected,
            checkpoint_sha256=digest(pending),checkpoint_size=pending.stat().st_size,
            archive_sha256=digest(archive),config_relative=rel,config_sha256=digest(config),
            source=SOURCE,architecture=version,degradation=family,
            provenance_limit='published author release; exact training history/regularization not independently verified')
        os.replace(pending,checkpoint)
        sidecar.write_text(json.dumps(record,indent=2)+'\n')
    return dict(id='author-'+release_id,label=f'Author LR-YOLO{version} · {"VOC_Dark" if family=="low_light" else "VOC_Haze"}',
        architecture=version,degradation=family,stage='dark',case_id=0,origin='author_release',
        checkpoint=str(checkpoint),config=str(config),sha256=record['checkpoint_sha256'],
        release_id=release_id,release_source=SOURCE)
