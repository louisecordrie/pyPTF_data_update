#!/bin/bash

#############################
# SETTINGS
#
# eventID: it controls the target event
#         existing events in IO:  2020_1030_samos, 2018_1025_zante
#
# domain: it controls the target area, target points, and relative settings
#         existing domains in IO:  med-tsumaps, pacific-cheese
#
#############################

eventID_list=[2003_0521_boumardes,2015_0416_crete,2015_1117_lefkada,2016_0125_gibraltar,2016_1030_norcia,2017_0612_lesbo,2017_0720_kos-bodrum,2018_1025_zante,2019_0320_turkey,2019_0921_albania,2020_0502_crete,2020_1030_samos,crete_synthetic_event,neamwave_synthetic_event,samos_synth_event,2023_0909_morocco]
eventID=2023_0206_turkey
domain=med-tsumaps
tsu_sim=precomputed 

step1YN=true
step2YN=false
step3YN=true

#############################
# CONFIGURATION
#############################
mainFolder=$(pwd)
alert_lev=true
workdir=$mainFolder/IO/$domain\_$eventID
echo $workdir
if [ ! -d $workdir ]; then
    mkdir $workdir
    mkdir $workdir/step3_output
    mkdir $workdir/step1_output
    chmod 775 $workdir
else
    echo 'Output file already exists: results will be overwritten'
fi
if [ ! -d $workdir/step3_output ]; then
    mkdir $workdir/step3_output
    chmod 775 $workdir/step3_output
else
    echo 'Output file already exists: results will be overwritten'
fi
if [ ! -d $workdir/step1_output ]; then
    mkdir $workdir/step1_output
    chmod 775 $workdir/step1_output
else
    echo 'Output file already exists: results will be overwritten'
fi
cp $mainFolder/IO/earlyEst/$eventID\_stat.json $workdir

#############################
# RUN SCRIPTS
#############################
echo 'Event:    '$eventID
echo 'Domain:   '$domain 
echo 'Run type: '$tsu_sim 
echo 'Workdir:  '$workdir 

#########
# STEP 1
#########
echo '============================'
echo '========== STEP 1 =========='
echo '============================'
if $step1YN; then
  echo 'run_step1.py --cfg ../cfg/ptf_main.config --event ../IO/earlyEst/'$eventID'_stat.json'
  cd $mainFolder/Step1_EnsembleDef_python
  python ./run_step1.py --cfg ../cfg/ptf_main.config --event ../IO/earlyEst/$eventID\_stat.json
  mv ptf_localOutput/* $workdir/step1_output/
  mv $workdir/step1_output/ptf_out*.hdf5 $mainFolder/Step3_HazardAggregation_python/ptf_localOutput/
else
   echo 'Not exectuted'  
fi

#########
# STEP 2
#########
echo ' '
echo ' '
echo '============================'
echo '========== STEP 2 =========='
echo '============================'
if $step2YN; then
   # for simulations on-the-fly
fi

#########
# STEP 3
# Example: matlab -nodisplay -nosplash -nodesktop -r "Step3_run('2020_1030_samos','med-tsumaps','precomputed'); exit;
#########
echo ' '
echo ' '
echo '============================'
echo '========== STEP 3 =========='
echo '============================'
if $step3YN; then
  cd $mainFolder/Step3_HazardAggregation_python
  echo '-------STEP 3-------'
  /home/louise.cordrie/MINICONDA/miniconda3/envs/env-miniconda/bin/python3.8 ./run_step3.py --cfg ../cfg/ptf_main.config --event ../IO/earlyEst/$eventID\_stat.json 
  mv $mainFolder/Step3_HazardAggregation_python/ptf_localOutput/* $workdir/step3_output/
  
else
   echo 'Not exectuted'  
fi
