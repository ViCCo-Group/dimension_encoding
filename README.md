# Distributed representations of behaviorally relevant object dimensions in the human visual system

This repository contains the analysis code used for the paper "[Distributed representations of behaviorally relevant object dimensions in the human visual system](https://doi.org/10.1101/2023.08.23.553812)." by O. Contier, Chris I. Baker, and Martin N. Hebart.


## Installation

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Data

The analyses relie on the openly available THINGS-fMRI and behavioral data [available on figshare](https://doi.org/10.25452/figshare.plus.c.6161151.v1). Further information on data download can be found in the [THINGS-data paper](https://doi.org/10.25452/figshare.plus.c.6161151.v1).

### Re-running analyses

Once the abovementioned data is installed into the `./data` directory, the analyses presented in the manuscript can be re-run by executing the respective python-scripts in `./scripts`.

While most of the analyses are feasible to run on a normal desktop computer or laptop, the parametric modulation model on the time series data (`run_pmod.py`) is memory intensive and likely only suited to run on computing clusters.