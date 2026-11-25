"""Opt-in exact predicate/gather fusion; no installed/default route change.

Build products must be placed in this experiment via TORCH_EXTENSIONS_DIR.
Torch retains the full de.max() operation. CPU use is deliberately rejected.
"""
import os
from pathlib import Path
import torch
_EXT=None

def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here=Path(__file__).parent
        _EXT=load(name='predicate_fusion_518',sources=[str(here/'predicate_fusion_518.cpp'),str(here/'predicate_fusion_518.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','-lineinfo'],verbose=True)
    return _EXT

def predicates(c,u,v,n,nab,half,aa,ae,points,penv,rho,ii,jj,gamma,debug=False):
    return extension().run([c,u,v,n,nab,half,aa,ae,points,penv,rho,ii,jj],float(gamma),debug)
