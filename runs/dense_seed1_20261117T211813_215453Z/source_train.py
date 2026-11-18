from pathlib import Path
from datetime import datetime, timezone
import json, time, random, subprocess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from .data import preparation, vocabulary, Demonstrations
from .models import Policy

def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total,count = 0.,0
    by_task = {}
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            images = {k:v.to(device) for k,v in batch['images'].items()}
            targets = batch['action'].to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            predictions = model(images,batch['text'].to(device))
            per_example = (predictions-targets).square().mean(1)
            loss = per_example.mean()
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
            total += loss.item()*len(targets)
            count += len(targets)
            for task,error in zip(batch['task'],per_example.detach().cpu().tolist()):
                entry = by_task.setdefault(task,[0.,0])
                entry[0] += error; entry[1] += 1
    return total/count,{k:v[0]/v[1] for k,v in by_task.items()}

def train(root, variant, seed=0, epochs=10, batch_size=32, lr=0.001, smoke=False):
    root = Path(root)
    manifest,norm,digest = preparation(root)
    vocab = vocabulary(manifest)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = {s:Demonstrations(root,manifest,norm,vocab,s) for s in ('train','validation')}
    if smoke:
        datasets = {s:Subset(d,list(range(min(32,len(d))))) for s,d in datasets.items()}
        epochs=1
    generator = torch.Generator().manual_seed(seed)
    loaders = {s:DataLoader(d,batch_size=batch_size,shuffle=s=='train',num_workers=0,
                           generator=generator if s=='train' else None) for s,d in datasets.items()}
    model = Policy(variant,len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = root/('smoke_runs' if smoke else 'runs')/f'{variant}_seed{seed}_{name}'
    run.mkdir(parents=True,exist_ok=False)
    config = {'architecture':'compact_attention_v1','variant':variant,'training_seed':seed,
              'epochs':epochs,'batch_size':batch_size,'learning_rate':lr,'gradient_clip_norm':10.,
              'tokens_per_camera':model.tokens_per_camera,'vocab':vocab,'manifest_sha256':digest,
              'parameters':sum(p.numel() for p in model.parameters()),'smoke_test':smoke,
              'torch_version':str(torch.__version__),'device':str(device),
              'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
              'language_encoder':'randomly initialized word embedding + GRU; no pretrained weights',
              'vision_encoder':'randomly initialized compact CNN; no pretrained weights'}
    config['git_revision'] = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    config['git_dirty'] = bool(subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True).stdout.strip())
    (run/'config.json').write_text(json.dumps(config,indent=2))
    # Capture implementation source so later edits cannot obscure what was run.
    for name in ('models.py','data.py','train.py'):
        (run/('source_'+name)).write_text((root/'t2'/name).read_text())
    (run/'split_manifest.json').write_bytes((root/'configs/data_preparation/split_manifest.json').read_bytes())
    (run/'action_normalization.json').write_text(json.dumps(norm,indent=2))
    history=[];best=float('inf')
    print('Run:',run,flush=True)
    for i in range(1,epochs+1):
        start=time.perf_counter()
        train_loss,_=epoch(model,loaders['train'],device,optimizer)
        val_loss,per_task=epoch(model,loaders['validation'],device)
        history.append({'epoch':i,'train_mse':train_loss,'validation_mse':val_loss,
                        'validation_by_task':per_task,'seconds':time.perf_counter()-start})
        (run/'history.json').write_text(json.dumps(history,indent=2))
        if val_loss<best:
            best=val_loss
            torch.save({'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'config':config,'normalization':norm,'epoch':i,'validation_mse':val_loss},run/'best_model.pt')
        print(f'Epoch {i}/{epochs}: train={train_loss:.5f}, validation={val_loss:.5f}',flush=True)
    restored=torch.load(run/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(restored['model_state_dict'])
    verified,per_task=epoch(model,loaders['validation'],device)
    assert np.isclose(verified,restored['validation_mse'],rtol=1e-4,atol=1e-6)
    (run/'validation_summary.json').write_text(json.dumps({'selected_epoch':restored['epoch'],
         'validation_mse':verified,'per_task':per_task,'checkpoint_reload_verified':True},indent=2))
    return run

def load_policy(checkpoint_path, device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if saved['config'].get('architecture')!='compact_attention_v1':
        raise ValueError('Use the existing 03 notebook for the legacy pilot checkpoint; this loader expects the new shared architecture.')
    model=Policy(saved['config']['variant'],len(saved['config']['vocab'])).to(device)
    model.load_state_dict(saved['model_state_dict']);model.eval()
    return model,saved,device
