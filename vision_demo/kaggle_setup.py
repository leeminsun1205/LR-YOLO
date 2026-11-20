"""Explicit notebook helpers. No download, install or model work at import."""
import hashlib
import json
import os
from pathlib import Path
from .methods import model_label


def discover_models(roots=('/kaggle/input',)):
    """Read small sidecars, never deserialize every mounted checkpoint."""
    result=[]
    for root in roots:
        if not Path(root).exists(): continue
        for folder,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in ('JPEGImages','Annotations','images','.git','node_modules','__pycache__') and not d.startswith('.')]
            if 'provenance.json' not in files or 'resolved_config.py' not in files:continue
            folder=Path(folder)
            try:
                p=json.loads((folder/'provenance.json').read_text());c=p['contract']
                if c['action']!='train':continue
                family=c.get('degradation','low_light');version=c.get('model_version','v10')
                stage=c['stage'];case=c['case_id']
                if family not in ('low_light','haze') or version not in ('v8','v10'):continue
                best=list(folder.glob('best_*.pth'))
                if len(best)>1:
                    print('Ambiguous best checkpoint; registry needs explicit selection:',folder,flush=True);continue
                checkpoint=best[0] if best else folder/'latest_resume.pth'
                if not checkpoint.is_file():continue
                state=json.loads((folder/'session_state.json').read_text()) if (folder/'session_state.json').is_file() else {}
                ident=f'{version}-{family}-{stage}-s{case}-'+hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:8]
                result.append(dict(id=ident,label=f'{model_label(version,case,stage)} · {folder.name[-6:]}',
                    architecture=version,degradation=family,stage=stage,case_id=case,
                    checkpoint=str(checkpoint),config=str(folder/'resolved_config.py'),
                    completed_epochs=state.get('completed_epochs'),run_target_epochs=state.get('target_epochs')))
            except (KeyError,ValueError,TypeError):continue
    return result


def prepare_gallery(family,cache,auto_download=True,overrides=None,pool='clean_val'):
    """Reuse immutable native manifests. Downloads only when explicitly called."""
    from selective_protocol_data import prepare_training,prepare_final
    if pool=='clean_val':
        datasets=prepare_training(overrides or {},cache,degradation=family,auto_download=auto_download)
        spec=datasets['clean_val'];degraded=False;source='VOC_Dark clean' if family=='low_light' else 'VOC_Haze clean'
    elif pool=='fixed_test':
        spec=prepare_final(overrides or {},cache,'voc',degradation=family,auto_download=auto_download)
        degraded=True;source='VOC_Dark_Val' if family=='low_light' else 'VOC_Haze_Val'
    else:raise ValueError('Gallery pool must be clean_val or fixed_test')
    return dict(id=f'{family}_{pool}',root=spec['root'],prefix=spec['prefix'],manifest=spec['manifest'],
                degradation=family,split=pool,source=source,already_degraded=degraded)


def configure_ngrok(ngrok, secrets_client=None, token=None):
    """Resolve credentials without printing them; no interactive prompt in batch runs."""
    token=token or os.environ.get('NGROK_AUTHTOKEN')
    if not token:
        try:
            if secrets_client is None:
                from kaggle_secrets import UserSecretsClient
                secrets_client=UserSecretsClient()
            token=secrets_client.get_secret('NGROK_AUTH_TOKEN')
        except Exception:
            raise RuntimeError(
                'Cannot read NGROK_AUTH_TOKEN from Kaggle Secrets. '
                'Check Add-ons > Secrets: exact label and permission for this notebook. '
                'The Kaggle Secrets backend may also be unavailable. '
                'For an interactive session only, set NGROK_AUTHTOKEN in memory '
                'using getpass, then rerun this server cell. No server was started.'
            ) from None
    if not isinstance(token,str) or not token.strip():
        raise ValueError('Ngrok token is empty')
    ngrok.set_auth_token(token.strip())
