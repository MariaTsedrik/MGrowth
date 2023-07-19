# MGrowth

MGrowth is a Python package for computing growth factor and rate in various cosmological scenarios.

## Installation (not pip-installable yet)

Use the package manager [pip](https://pypi.org) to install MGrowth.

```bash
pip install MGrowth
```

## Requirements
Required python packages:
* numpy
* scipy


## Available Models
- LCDM: standard cosmological model with fixed $w=-1$
- wCDM: dark energy with constant $w$ which can take any values
- w0waCDM:  [CPL evolving dark energy](https://arxiv.org/abs/gr-qc/0009008) $w = w_0 + (1-a)w_a$
- IDE: Interacting Dark Energy also known as [dark scattering](https://arxiv.org/abs/1605.05623)
- nDGP: normal branch of [DGP gravity](https://arxiv.org/abs/hep-th/0005016)
- f(R):  [Hu-Sawicki f(R)](https://arxiv.org/abs/0705.1158)
- Linder Gamma: [Growth index parametrisation](https://arxiv.org/abs/astro-ph/0507263)
- Linder Gamma a: [Time-dependent growth index](https://arxiv.org/abs/2304.07281)

## Documentation
Documentation is available in docs/_build/html/index.html
