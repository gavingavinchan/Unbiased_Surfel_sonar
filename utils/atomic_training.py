"""Atomic generations for explicitly owned training state.

This is a serializer, not a claim of native CUDA release/resume equivalence.
The caller must include every enabled strategy and sampler in its state.
Only torch's restricted weights-only decoder is used; source/config binding and
payload checksum are checked before decoding. Generations are never overwritten.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch

REQUIRED = {'step', 'model', 'optimizers', 'schedulers', 'sampler', 'strategy', 'scale', 'rng'}


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {'python': random.getstate(),
        'numpy': [name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    name, keys, pos, has_gauss, cached = state['numpy']
    if len(state['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA RNG device count differs')
    random.setstate(state['python'])
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()): raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for x in value.values(): finite_tree(x)
    elif isinstance(value, (tuple, list)):
        for x in value: finite_tree(x)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite checkpoint scalar')


def validate(state):
    if set(state) != REQUIRED: raise ValueError('incomplete or unknown training state')
    if type(state['step']) is not int or state['step'] < 0: raise ValueError('invalid step')
    if set(state['rng']) != {'python','numpy','torch','cuda'}: raise ValueError('incomplete RNG state')
    if not isinstance(state['sampler'], dict) or not isinstance(state['strategy'], dict):
        raise ValueError('sampler/strategy state must be explicit dictionaries')
    finite_tree(state)


def identity(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save(directory, state, binding):
    validate(state)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    filename = f"generation-{state['step']:07d}-{uuid.uuid4().hex}.pt"
    payload = directory / filename
    with payload.open('xb') as f:
        torch.save({'schema': 1, 'binding': binding, 'state': state}, f)
        f.flush(); os.fsync(f.fileno())
    manifest = {'schema': 1, 'file': filename, 'step': state['step'],
        'sha256': hashlib.sha256(payload.read_bytes()).hexdigest(), 'binding_sha256': identity(binding)}
    tmp = directory / ('latest-' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(manifest, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    sync_directory(directory)
    os.replace(tmp, directory / 'latest.json')
    sync_directory(directory)
    return manifest


def load(directory, binding, device='cpu'):
    directory = Path(directory)
    manifest = json.loads((directory / 'latest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('binding_sha256') != identity(binding):
        raise ValueError('checkpoint source/config binding differs')
    name = manifest['file']
    if Path(name).name != name or not name.startswith('generation-') or not name.endswith('.pt'):
        raise ValueError('invalid generation path')
    path = directory / name
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('checkpoint payload checksum differs')
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('schema') != 1 or payload.get('binding') != binding:
        raise ValueError('checkpoint payload binding differs')
    state = payload['state']; validate(state)
    if state['step'] != manifest['step']: raise ValueError('checkpoint step differs')
    return state
