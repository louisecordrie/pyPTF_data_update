import os
import sys
import ray
import h5py
import math
import scipy
import numpy as np
import pathlib
import hickle as hkl
import json
import pprint
import glob
from time              import gmtime
from datetime          import datetime
from numba             import jit
from scipy.stats       import lognorm
from ismember          import ismember
from numba_stats import norm
import configparser
from ptf_mix_utilities import ccc
from ptf_parser              import parse_ptf_stdin
from ptf_parser              import update_cfg
from ptf_parser                import parse_ptf_stdin
from ptf_parser                import update_cfg
from ptf_mix_utilities         import conversion_to_utm
from ptf_preload             import load_PSBarInfo
from ptf_preload             import ptf_preload
from ptf_preload             import load_Scenarios_Reg
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import parse_ptf_stdin
from ptf_parser              import update_cfg
from ptf_figures             import make_ptf_figures
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_as_txt
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import save_ptf_out
from ptf_mix_utilities        import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_hazard_curves           import define_thresholds
from ptf_hazard_curves           import compute_generic_hazard_curve_threshold
from ptf_preload                 import load_scenarios
from ptf_preload_curves          import reallocate_curves
from scipy.stats import vonmises
from scipy.stats import expon
from scipy.stats import norm
from copy import deepcopy


####################################
# Begin
####################################

def compute_ensemble_mare(**kwargs):

    Config                = kwargs.get('cfg', None)
    ee                    = kwargs.get('event_parameters', None)
    args                  = kwargs.get('args', None)
    LongTermInfo          = kwargs.get('LongTermInfo', None)
    POIs                  = kwargs.get('pois', None)
    hazard_curves_files   = kwargs.get('h_curve_files', None)
    ptf_out               = kwargs.get('ptf_out', None)
    type_ens              = kwargs.get('type_ens', None)
    samp_test             = kwargs.get('samp_test', None)
    samp_weight           = kwargs.get('samp_weight', None)
    tg                    = kwargs.get('tsu_data', None) 

    name_type_ens = 'new_ensemble_'+type_ens

    for Nid in range(len(ptf_out['new_ensemble_'+type_ens])):

        probability_scenarios=ptf_out[name_type_ens][Nid]
       

        len_scen = len(probability_scenarios['ptf_val'])
        # Initialisation of the obs-mod comparison dict
        final_logk=np.zeros(len_scen)
        final_logK=np.zeros(len_scen)
        NRMS=np.zeros(len_scen)
        NRMSE=np.zeros(len_scen)
        sigmasimu=1.0
        datj_all = probability_scenarios['ptf_val']
        orig_all = probability_scenarios['obs_val']
        omean = np.mean(orig_all)
        tg = probability_scenarios['obs_val'][0]
        n=float(len(tg))

        for i in range(len_scen): #scenario
            logk=0.0
            logK=0.0
            nrmsd=0.0
            nrmso=0.0
            for j in range(len(tg)): #pois
                datj = datj_all[i,j]
                orig = orig_all[i,j]
                if datj<0.001:
                   datj=0.001
                if orig<0.001:
                   orig=0.001
                nrmsd=nrmsd+(orig-datj)**2
                nrmso=nrmso+(orig-omean)**2
                if nrmso < 0.000001:
                   nrmso = 0.000001

            ### Choice of the metric to be used for reweighting ###
            NRMS[i]=math.sqrt(nrmsd)/math.sqrt(nrmso)
            #NRMSE[i]=math.sqrt(nrmsd/len(origin))/np.amax(origin) 
            ptf_out[name_type_ens][Nid]['NRMS'][i]=NRMS[i] #final_logK[i]

        mare_weight_tmp = np.zeros((len(ptf_out[name_type_ens][Nid]['NRMS'])))
        mare_weight_tmp = expon.pdf(NRMS,loc=0,scale=0.5)
        mare_weight     = mare_weight_tmp/np.sum(mare_weight_tmp)
        ptf_out[name_type_ens][Nid]['mare_proba']=mare_weight


        if len(probability_scenarios['ProbScenBS'])>1 and len(probability_scenarios['ProbScenPS'])>1:
           ProbScenBS_all=ptf_out[name_type_ens][Nid]['ProbScenBS']
           ProbScenBS_temp=ProbScenBS_all*mare_weight[0:len(ProbScenBS_all)]
           ProbScenPS_all=ptf_out[name_type_ens][Nid]['ProbScenPS']
           ProbScenPS_temp=ProbScenPS_all*mare_weight[len(ProbScenBS_all)::]
           totprob=np.sum(ProbScenPS_temp)+np.sum(ProbScenBS_temp)
           ptf_out[name_type_ens][Nid]['ProbScenPS']=ProbScenPS_temp/totprob
           ptf_out[name_type_ens][Nid]['ProbScenBS']=ProbScenBS_temp/totprob
        
        elif len(probability_scenarios['ProbScenBS'])>1:
           ProbScenBS_all=ptf_out[name_type_ens][Nid]['ProbScenBS']
           ProbScenBS_temp=ProbScenBS_all*mare_weight
           ptf_out[name_type_ens][Nid]['ProbScenBS']=ProbScenBS_temp/np.sum(ProbScenBS_temp)

        elif len(probability_scenarios['ProbScenPS'])>1:
           ProbScenPS_all=ptf_out[name_type_ens][Nid]['ProbScenPS']
           ProbScenPS_temp=ProbScenPS_all*mare_weight
           ptf_out[name_type_ens][Nid]['ProbScenPS']=ProbScenPS_temp/np.sum(ProbScenPS_temp)

    return ptf_out
