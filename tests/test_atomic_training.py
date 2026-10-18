import copy
import json
import random
import numpy as np
import pytest
import torch
from utils.atomic_training import load, save, rng_state, restore_rng


def fixture():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    order = list(range(7)); random.shuffle(order)
    return model, optimizer, scheduler, {'order':order,'cursor':0}, []


def advance(model, optimizer, scheduler, sampler, draws, first, last):
    for step in range(first, last):
        if sampler['cursor'] == len(sampler['order']):
            random.shuffle(sampler['order']); sampler['cursor'] = 0
        camera = sampler['order'][sampler['cursor']]; sampler['cursor'] += 1
        draws.append(camera)
        x = torch.randn(2) + float(np.random.rand()) + random.random() + camera
        model(x).square().sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()


def snapshot(items, step):
    m,o,s,sampler,_ = items
    return {'model':m.state_dict(),'optimizers':o.state_dict(),'schedulers':s.state_dict(),
        'step':step,'sampler':copy.deepcopy(sampler),'strategy':{'pruning':False},
        'scale':{'fixed':1.},'rng':rng_state()}


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def test_cpu_200_split_continuation(tmp_path):
    uninterrupted = fixture(); advance(*uninterrupted,0,200)
    expected = snapshot(uninterrupted,200)
    split = fixture(); advance(*split,0,100)
    binding={'source':'fixture-source','config':{'lr':.01}}
    save(tmp_path,snapshot(split,100),binding)
    actual = fixture() # destroy and recreate all model/optimizer/sampler objects
    state = load(tmp_path,binding)
    m,o,s,sampler,draws = actual
    m.load_state_dict(state['model']);o.load_state_dict(state['optimizers']);s.load_state_dict(state['schedulers'])
    sampler.update(state['sampler']);restore_rng(state['rng'])
    advance(*actual,state['step'],200)
    equal(expected,snapshot(actual,200))
    assert uninterrupted[-1][100:]==draws


def test_source_and_corruption_refused(tmp_path):
    state=snapshot(fixture(),0);save(tmp_path,state,{'source':'a'})
    with pytest.raises(ValueError,match='binding'):load(tmp_path,{'source':'b'})
    manifest=json.loads((tmp_path/'latest.json').read_text())
    (tmp_path/manifest['file']).write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):load(tmp_path,{'source':'a'})


def test_incomplete_and_nonfinite_refused(tmp_path):
    state=snapshot(fixture(),0);del state['rng']
    with pytest.raises(ValueError,match='incomplete'):save(tmp_path,state,{})
    state=snapshot(fixture(),0);state['model']['weight'].fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):save(tmp_path,state,{})


def test_interrupted_pointer_retains_previous_generation(tmp_path,monkeypatch):
    import utils.atomic_training as module
    state=snapshot(fixture(),0);manifest=save(tmp_path,state,{})
    def interrupt(*args):raise OSError('simulated interruption before pointer replacement')
    monkeypatch.setattr(module.os,'replace',interrupt)
    with pytest.raises(OSError):save(tmp_path,snapshot(fixture(),100),{})
    assert load(tmp_path,{})['step']==0
    assert json.loads((tmp_path/'latest.json').read_text())==manifest
