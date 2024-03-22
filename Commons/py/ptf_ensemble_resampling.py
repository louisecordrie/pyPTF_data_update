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

def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]


def mc_samp(ptf_out,name_type_ens,type_ens,MC_samp_scen): 

    Nid=0
    ptf_out[name_type_ens]=dict()
    ptf_out[name_type_ens][Nid]=dict()
    probability_scenarios = ptf_out['new_ensemble_MC'][Nid]
    Nsamp=MC_samp_scen
 
    if len(probability_scenarios['ProbScenBS'])>1 and len(probability_scenarios['ProbScenPS'])>1:
       
        print('Samp both PS and BS')
 
        perbs=len(probability_scenarios['ProbScenBS'])/(len(probability_scenarios['ProbScenBS'])+len(probability_scenarios['ProbScenPS']))
        NsampBS=int(perbs*Nsamp)
        NsampPS=Nsamp-NsampBS
        ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'] = np.zeros( (NsampBS,  5) )
        ptf_out[name_type_ens][Nid]['prob_scenarios_bs'] = np.zeros( (NsampBS) )
        ptf_out[name_type_ens][Nid]['real_prob_scenarios_bs'] = np.zeros( (NsampBS) )
        ptf_out[name_type_ens][Nid]['par_scenarios_bs'] = np.zeros(  (NsampBS, 11) )
        ptf_out[name_type_ens][Nid]['iscenbs']=np.zeros(NsampBS)
        ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'] = np.zeros( (NsampPS,  5) )
        ptf_out[name_type_ens][Nid]['prob_scenarios_ps'] = np.zeros( (NsampPS) )
        ptf_out[name_type_ens][Nid]['real_prob_scenarios_ps'] = np.zeros( (NsampPS) )
        ptf_out[name_type_ens][Nid]['par_scenarios_ps'] = np.zeros(  (NsampPS, 7) )
        ptf_out[name_type_ens][Nid]['iscenps']=np.zeros(NsampPS)

        ### BS ###
        int_ens = np.zeros(len(probability_scenarios['ProbScenBS']))
        prob_cum = 0
        for i in range(len(probability_scenarios['ProbScenBS'])):
            prob_cum=prob_cum+probability_scenarios['ProbScenBS'][i]
            int_ens[i]= prob_cum
        random_value = np.random.random(NsampBS)
        iscenbs=0
        for i in random_value:
            ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
            idx,proba = find_nearest(int_ens,i)
            ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
            ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
            ptf_out[name_type_ens][Nid]['iscenbs'][iscenbs]=idx
            ptf_out[name_type_ens][Nid]['prob_scenarios_bs'][iscenbs]=probability_scenarios['ProbScenBS'][idx]
            for j in range(5):
                ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'][iscenbs,j]=probability_scenarios['prob_scenarios_bs_fact'][idx,j]
            for j in range(11):
                ptf_out[name_type_ens][Nid]['par_scenarios_bs'][iscenbs,j]=probability_scenarios['par_scenarios_bs'][idx,j]
            iscenbs=iscenbs+1
    
        ### PS ###
        int_ens = np.zeros(len(probability_scenarios['ProbScenPS']))
        prob_cum = 0
        for i in range(len(probability_scenarios['ProbScenPS'])):
            prob_cum=prob_cum+probability_scenarios['ProbScenPS'][i]
            int_ens[i]= prob_cum
        random_value = np.random.random(NsampPS)
        iscenps=0
        for i in random_value:
            ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
            idx,proba = find_nearest(int_ens,i)
            ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
            ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
            ptf_out[name_type_ens][Nid]['iscenps'][iscenps]=idx
            ptf_out[name_type_ens][Nid]['prob_scenarios_ps'][iscenps]=probability_scenarios['ProbScenPS'][idx]
            for j in range(5):
                ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'][iscenps,j]=probability_scenarios['prob_scenarios_ps_fact'][idx,j]
            for j in range(7):
                ptf_out[name_type_ens][Nid]['par_scenarios_ps'][iscenps,j]=probability_scenarios['par_scenarios_ps'][idx,j]
            iscenps=iscenps+1

        # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
        ptf_out[name_type_ens][Nid]['nr_bs_scenarios'] = np.shape(ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'])[0]
        ProbScenBS        = ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'].prod(axis=1)
        ptf_out[name_type_ens][Nid]['nr_ps_scenarios'] = np.shape(ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'])[0]
        ProbScenPS        = ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'].prod(axis=1)
        TotBS=len(ProbScenBS)
        TotPS=len(ProbScenPS)
        Tot=len(ProbScenBS)+len(ProbScenPS)
        ProbScenBS = np.ones(TotBS)/Tot
        ProbScenPS = np.ones(TotPS)/Tot
        ptf_out[name_type_ens][Nid]['ProbScenBS'] = ProbScenBS
        ptf_out[name_type_ens][Nid]['relevant_scenarios_bs'] = np.unique(ptf_out[name_type_ens][Nid]['par_scenarios_bs'][:,0])
        ptf_out[name_type_ens][Nid]['ProbScenPS'] = ProbScenPS
        ptf_out[name_type_ens][Nid]['relevant_scenarios_ps'] = np.unique(ptf_out[name_type_ens][Nid]['par_scenarios_ps'][:,0])
    
    elif len(probability_scenarios['ProbScenBS'])>1:

           NsampBS=Nsamp
           ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'] = np.zeros( (NsampBS,  5) )
           ptf_out[name_type_ens][Nid]['prob_scenarios_bs'] = np.zeros( (NsampBS) )
           ptf_out[name_type_ens][Nid]['real_prob_scenarios_bs'] = np.zeros( (NsampBS) )
           ptf_out[name_type_ens][Nid]['par_scenarios_bs'] = np.zeros(  (NsampBS, 11) )
           ptf_out[name_type_ens][Nid]['iscenbs']=np.zeros(NsampBS)

           ### BS ###
           print('Sampling for BS only')
           int_ens = np.zeros(len(probability_scenarios['ProbScenBS']))
           prob_cum = 0
           for i in range(len(probability_scenarios['ProbScenBS'])):
               prob_cum=prob_cum+probability_scenarios['ProbScenBS'][i]
               int_ens[i]= prob_cum
           random_value = np.random.random(NsampBS)
           iscenbs=0
           for i in random_value:
               ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
               idx,proba = find_nearest(int_ens,i)
               ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
               ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
               ptf_out[name_type_ens][Nid]['iscenbs'][iscenbs]=idx
               ptf_out[name_type_ens][Nid]['prob_scenarios_bs'][iscenbs]=probability_scenarios['ProbScenBS'][idx]
               for j in range(5):
                   ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'][iscenbs,j]=probability_scenarios['prob_scenarios_bs_fact'][idx,j]
               for j in range(11):
                   ptf_out[name_type_ens][Nid]['par_scenarios_bs'][iscenbs,j]=probability_scenarios['par_scenarios_bs'][idx,j]
               iscenbs=iscenbs+1


           # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
           ptf_out[name_type_ens][Nid]['nr_bs_scenarios'] = np.shape(ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'])[0]
           ProbScenBS        = ptf_out[name_type_ens][Nid]['prob_scenarios_bs_fact'].prod(axis=1)
           TotProbBS_preNorm = np.sum(ProbScenBS)
           TotBS=len(ProbScenBS)
           ProbScenBS = np.ones(TotBS)/TotBS
           ptf_out[name_type_ens][Nid]['ProbScenBS'] = ProbScenBS
           ptf_out[name_type_ens][Nid]['relevant_scenarios_bs'] = np.unique(ptf_out[name_type_ens][Nid]['par_scenarios_bs'][:,0])
           ptf_out[name_type_ens][Nid]['ProbScenPS'] = []

    else: #len(probability_scenarios['ProbScenPS']):

           NsampPS=Nsamp
           ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'] = np.zeros( (NsampPS,  5) )
           ptf_out[name_type_ens][Nid]['prob_scenarios_ps'] = np.zeros( (NsampPS) )
           ptf_out[name_type_ens][Nid]['real_prob_scenarios_ps'] = np.zeros( (NsampPS) )
           ptf_out[name_type_ens][Nid]['par_scenarios_ps'] = np.zeros(  (NsampPS, 7) )
           ptf_out[name_type_ens][Nid]['iscenps']=np.zeros(NsampPS)

           ### PS ###
           print('Sampling for PS only')
           int_ens = np.zeros(len(probability_scenarios['ProbScenPS']))
           prob_cum = 0
           for i in range(len(probability_scenarios['ProbScenPS'])):
               prob_cum=prob_cum+probability_scenarios['ProbScenPS'][i]
               int_ens[i]= prob_cum
           random_value = np.random.random(NsampPS)
           iscenps=0
           for i in random_value:
               ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
               idx,proba = find_nearest(int_ens,i)
               ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
               ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
               ptf_out[name_type_ens][Nid]['iscenps'][iscenps]=idx
               ptf_out[name_type_ens][Nid]['prob_scenarios_ps'][iscenps]=probability_scenarios['ProbScenPS'][idx]
               for j in range(5):
                   ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'][iscenps,j]=probability_scenarios['prob_scenarios_ps_fact'][idx,j]
               for j in range(7):
                   ptf_out[name_type_ens][Nid]['par_scenarios_ps'][iscenps,j]=probability_scenarios['par_scenarios_ps'][idx,j]
               iscenps=iscenps+1

           # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
           ptf_out[name_type_ens][Nid]['nr_ps_scenarios'] = np.shape(ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'])[0]
           ProbScenPS        = ptf_out[name_type_ens][Nid]['prob_scenarios_ps_fact'].prod(axis=1)
           TotProbPS_preNorm = np.sum(ProbScenPS)
           TotPS=len(ProbScenPS)
           ProbScenPS = np.ones(TotPS)/TotPS
           ptf_out[name_type_ens][Nid]['ProbScenPS'] = ProbScenPS
           ptf_out[name_type_ens][Nid]['relevant_scenarios_ps'] = np.unique(ptf_out[name_type_ens][Nid]['par_scenarios_ps'][:,0])
           ptf_out[name_type_ens][Nid]['ProbScenBS'] = []

    return ptf_out
