# README for pyPTF_data_update v1.0

## Probabilistic Tsunami Forecast Data Update

The Probabilistic Tsunami Forecast (PTF) is an ensemble-based approach that aggregates multiple tsunami simulations, covering the uncertainty associated with seismic sources. Post-processing also considers uncertainty in tsunami propagation. The code has been validated on multiple test cases. Results and methods can be found in the following article:

Selva, J., Lorito, S., Volpe, M., Romano, F., Tonini, R., Perfetti, P., Bernardi, F., Taroni, M., Scala, A., Babeyko, A., Løvholt, F., Gibbons, S.J., Macías, J., Castro, M.J., González-Vida, J.M., Sánchez-Linares, C., Bayraktar, H.B., Basili, R., Maesano, F.E., Tiberti, M.M., Mele, F., Piatanesi A., & Amato, A. Probabilistic tsunami forecasting for early warning. Nat Commun 12, 5677 (2021). [DOI: 10.1038/s41467-021-25815-w](https://doi.org/10.1038/s41467-021-25815-w)

pyPTF_data_update is derived from the PTF and is written in Python with various implemented functionalities. The workflow consists of two main steps:
- Creating an ensemble of scenarios based on given seismic sources and user parameters
- Post-processing the results and calculating forecasts

Details on this workflow can be found in the following article:

## pyPTF_data_update Code

This repository contains all the source codes used to run pyPTF_data_update in a local environment. Follow the instructions in the RUN section to use the workflow.

The code has been developed and tested in Python version 3.7.

The entire workflow can be run using the script `run_py.sh`, and the setup of the environment and configuration of the run are described in the following sections.

## Preparation of the Environment

Download the entire git repository. A dataset should be uploaded from Zenodo and uncompressed. Replace line 36 in the `cfg/ptf_main.config` file with your `pyPTF_hazard_curves` dictionary path.

## Configuration of the Run

Two files must be modified before the run:

- `cfg/ptf_main.config`
- `run_py.sh`

### ptf_main.config
Modify the [Sampling] section (line 129) and the [EventID] section (line 152) as described in the file.

### run_py.sh
Modify the eventID (line 15) to match the one in `ptf_main.config`. Optionally, modify the Python version.

### Run
Execute the following command:
$ bash run_py.sh
