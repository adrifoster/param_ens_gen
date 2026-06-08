# Parameter Ensemble Examples

These examples demonstrate how to use `param_ens_gen` to generate parameter ensembles
for either CLM (`examples/clm_example`) or FATES (*coming soon*) for CLM and/or FATES.

It includes two examples: a Latin Hypercube ensemble and a One-at-a-time ensemble.

## Contents

### CLM Example

```text
clm_example/
    clm_param_dir/       parameter metadata (main.csv and per-PFT bound files)
    default_clm_param.nc default CLM parameter file
    lh_config.yaml       Latin Hypercube ensemble configuration
    oat_config.yaml      One-at-a-time ensemble configuration
```

## Setup

From the repo root, install the package and activate your environment:

```bash
conda activate param_ens_gen
pip install -e .
```

## Running the examples

All commands should be run from inside the `examples/clm_example/` or `examples/fates_example` directory:

```bash
cd examples/clm_example
```

### Latin Hypercube ensemble

Generates 20 ensemble members by sampling all parameters simultaneously using a Latin Hypercube design.

```bash
param_ens_gen run lh_config.yaml
```

Output it written to `output_LH/`. This produces:

* `clm_lh_000.nc` through `clm_lh_019.nc`: one parameter file per ensemble member
* `clm_lh_key.csv`: normalized parameter values for each member
* `clm_lh.txt`: list of ensemble member names, for use in scripting

### One-at-a-time ensemble

Generates two ensemble members per parameter: one at its minimum value and one at its maximum value. All other parameters are held at default.

```bash
param_ens_gen run oat_config.yaml
```

Output is written to `output_OAT/`. This produces:

* `clm_lh_000.nc` through `clm_lh_00n.nc`: One parameter file per parameter per direction
* `clm_oat_key.csv`: parameter name and direction for each member
* `clm_oat.txt`: list of ensemble member names

## Parameters

All parameters defined in in main.csv's are varied in all ensembles. See the main `param_ens_gen` README for a full description of the parameter metadata format and supported parameter types.

