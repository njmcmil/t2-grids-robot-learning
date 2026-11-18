"""Meaningful smoke checks: actual data, gradients through coordinates, and language controls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,torchvision
from torch.utils.data import DataLoader
from t2.data import preparation,vocabulary,Demonstrations
from t2.models import Policy,VARIANTS
manifest,norm,digest=preparation(ROOT);vocab=vocabulary(manifest)
data=Demonstrations(ROOT,manifest,norm,vocab,'train')
batch=next(iter(DataLoader(data,batch_size=2)))
for variant in VARIANTS:
    torch.manual_seed(0);model=Policy(variant,len(vocab))
    prediction,coords=model(batch['images'],batch['text'],return_coordinates=True)
    assert prediction.shape==(2,7) and torch.isfinite(prediction).all()
    for grid in coords.values():assert grid.shape==(2,model.tokens_per_camera,2) and grid.abs().max()<=1
    loss=(prediction-batch['action']).square().mean();loss.backward()
    assert model.encoder.layers[0].weight.grad.abs().sum()>0
    if variant in ('grids','grids_film'):
        assert model.coordinate_predictor[-1].weight.grad.abs().sum()>0, 'Sampling gradients did not reach coordinates'
    if variant=='grids_film':assert model.film.weight.grad.abs().sum()>0
    swapped=batch['text'].clone()
    swapped[swapped==vocab['plate']]=vocab['stove']
    with torch.no_grad():
        changed=model(batch['images'],swapped)
        if variant=='vision_only':assert torch.equal(prediction.detach(),changed)
        else:assert not torch.allclose(prediction.detach(),changed,atol=1e-7,rtol=0)
    print(variant,'passed: finite actions, backbone gradients, token budget, language control',flush=True)
print('All model checks passed. These checks do not establish task success.')
