# README for pyPTF_data_update v1.0

## Probabilistic Tsunami Forecast Data Update

The Probabilistic Tsunami Forecast (PTF) is an ensemble-based approach that aggregates multiple tsunami simulations, covering the uncertainty associated with seismic sources. Post-processing also considers uncertainty in tsunami propagation. The code has been validated on multiple test cases. Results and methods can be found in the following article:

Selva, J., Lorito, S., Volpe, M., Romano, F., Tonini, R., Perfetti, P., Bernardi, F., Taroni, M., Scala, A., Babeyko, A., Løvholt, F., Gibbons, S.J., Macías, J., Castro, M.J., González-Vida, J.M., Sánchez-Linares, C., Bayraktar, H.B., Basili, R., Maesano, F.E., Tiberti, M.M., Mele, F., Piatanesi A., & Amato, A. Probabilistic tsunami forecasting for early warning. Nat Commun 12, 5677 (2021). [DOI: 10.1038/s41467-021-25815-w](https://doi.org/10.1038/s41467-021-25815-w)

pyPTF_data_update is derived from the PTF and is written in Python with various implemented functionalities. The workflow consists of two main steps:
- Creating an ensemble of scenarios based on given seismic sources and user parameters
- Post-processing the results and calculating forecasts

Details on this workflow can be found in the following article:

## Installation

This repository contains all the source codes used to run pyPTF_data_update in a local environment. Follow the instructions in the RUN section to use the workflow.

The code has been developed and tested in Python version 3.7.

The entire workflow can be run using the script `run_py.sh`, and the setup of the environment and configuration of the run are described in the following sections.

## Preparation of the Environment

Download the entire git repository. 

A dataset should be uploaded from Zenodo and uncompressed. 

Replace line 36 in the `cfg/ptf_main.config` file with your `pyPTF_hazard_curves` dictionary path.

Two files must be modified before the run:

- `cfg/ptf_main.config`
- `run_py.sh`

### ptf_main.config

The section `[Sampling]` (line 129) should be modified:

- The original scenario's ensemble can be used to produce the hazard curves with `OR_EM=1` (EM=ensemble) and `OR_HC=1` (HC=hazard curves): these options will produce two outputs ptf_out.hdf5 and hazard_curves_original.hdf5

- The resampling mode (`MC_type`, `MC_samp_scen`, `MC_samp_run`) is the one that should be used for the updating procedure:
  
  `MC_type=LH` (or MC) corresponds to the sampling method LatinHypercude or MonteCarlo
  
  `MC_samp_scen=500` corresponds to the number of scenarios in the ensemble
  
  `MC_samp_run=1` corresponds to the number of ensembles, it should not be modified (exists for development purposes)

- The RS options should be left to 0 (in development)
  
- The different updates can be activated with:
  
  `Kagan_weights=1` (1: activated, 0: not) for the focal mechanism update
  
  `Mare_weights=1` (1: activated, 0: not) for the tsunami data update
  
  `NbrFM=2` allows to select the number of focal-mechanism to be used in the update
  
- The rest of the options should not be modified
  

The section `[EventID]` (line 152) should be modified:

- The name of the event eventID can be modified using the exact same names as indicated in eventID_list (line 156)

  `eventID=2018_1025_zante`

### run_py.sh

Modify the eventID (line 15) to match the one in `ptf_main.config`. 

Optionally, modify the Python version (line 69 and line 100)

## Run

Execute the following command:
$ bash run_py.sh

## Post-Processing



## Test-Case



