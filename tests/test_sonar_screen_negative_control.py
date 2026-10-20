"""172 is an expected eligibility failure, never an accepted derivative case."""
import json
from pathlib import Path
import pytest
import torch
import negative_screen_172 as old

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_preserved_172_allocation_failure_signature(dtype):
    saved=json.loads((Path(__file__).parent/'fixtures/screen_center_172_negative_signatures.json').read_text())
    key=str(dtype).split('.')[-1]
    for case in old.cases():
        h=old.pack(case,dtype);signature=old.tile_signature(h)
        pixels,_=old.observations(case,dtype)
        observed=[]
        for j,eps in enumerate(old.CONFIG[key]['matrix_steps']):
            delta=torch.zeros_like(h);delta.reshape(-1)[j]=eps
            for sign,hh in [('plus',h+delta),('minus',h-delta)]:
                if old.tile_signature(hh)!=signature:
                    with pytest.raises(AssertionError):old.eligible(hh,pixels,signature)
                    observed.append(dict(label=f'matrix_{j}_{sign}',base_tile=list(signature),actual_tile=list(old.tile_signature(hh))))
        expected=next(r for r in saved if r['dtype']==key and r['case']==list(case))
        assert observed
        assert observed==[{k:r[k] for k in ['label','base_tile','actual_tile']} for r in expected['eligibility_failures']]
