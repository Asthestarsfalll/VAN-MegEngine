# VAN-MegEngine

The MegEngine Implementation of VAN(Visual Attention Network).

## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/ (Only support to van_03)

```bash
python convert_weights.py -m van_01
```

Import from megengine.hub:

Way 1:

```python
from functools import partial
import megengine.module as M
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/van-megengine:main', git_host='github.com')

# load VAN model and custom on you own
van = modelhub.VAN(embed_dims=[32, 64, 160, 256], mlp_ratios=[
    8, 8, 4, 4], norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2])

# load pretrained model
pretrained_model = modelhub.van_b0(pretrained=True)

```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'van_b0'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/van-megengine:main', entry=model_name, git_host='github.com', pretrained=True)
```

Currently support van_b0, van_b1, van_b2 , bat you can run convert_weights.py to convert other models(Due to official repo only offer s from van_b0 to van_b3, so the others don't )
For example:

```bash
  python convert_weights.py -m van_b3
```

Then load state dict manually.

```python
model = modelhub.van_b3()
model.load_state_dict(mge.load('./pretrained/van_b3.pkl'))
# or
model_name = 'van_b3'
model = hub.load(
    repo_info='asthestarsfalll/van-megengine:main', entry=model_name, git_host='github.com')
model.load_state_dict(mge.load('./pretrained/van_b3.pkl'))
```

## TODO

- [ ] add train codes maybe
- [ ] down stream tasks maybe

## Reference

[The official implementation of VAN](https://github.com/Visual-Attention-Network/VAN-Classification)
