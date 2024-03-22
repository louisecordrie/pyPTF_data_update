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

def extract_hmax(**kwargs):

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
    name_type_ens         = kwargs.get('name_type_ens', None)
    
    if name_type_ens == 'ori_ensemble':
       nrun = 1
    else:
       nrun = len(ptf_out[name_type_ens])
 
    for Nid in range(nrun):

        if name_type_ens == 'ori_ensemble':
           probability_scenarios=ptf_out['probability_scenarios']
        else:          
           probability_scenarios=ptf_out[name_type_ens][Nid]
            
        #### POIs ###
        idx=0
        selection=POIs['selected_pois']
        name=POIs['name']
        lon=np.zeros((len(selection)))
        lat=np.zeros((len(selection)))
        for i in range(len(selection)):
            idx=name.index(selection[i])
            lon[i]=POIs['lon'][idx]
            lat[i]=POIs['lat'][idx]
    
        ### Load scenarios
        scenarios_py_folder  =  Config.get('pyptf',  'Scenarios_py_Folder')
        py_bs_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListBS*npz'))
        Scenarios_BS = load_scenarios(list_scenarios=py_bs_scenarios, type_XS='BS', cfg=Config)
        py_ps_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListPS*npz'))
        Scenarios_PS = load_scenarios(list_scenarios=py_ps_scenarios, type_XS='PS', cfg=Config)
    
        ### Search the corresponding results
        type_of_measure          = args.intensity_measure
        int_meas_bs              = [] #np.zeros((len(selection),len(probability_scenarios['ProbScenBS'])))
        int_meas_ps              = [] #np.zeros((len(selection),len(probability_scenarios['ProbScenPS'])))
    
        if len(probability_scenarios['ProbScenBS'])>1:
    
           #print('Prob BS',probability_scenarios['ProbScenBS'])
           int_meas_bs              = np.zeros((len(selection),len(probability_scenarios['ProbScenBS'])))
           vec                      = np.array([100000000,100000,100,1,0.0100,1.0000e-04,1.0000e-06])
           relevant_scenarios_bs    = probability_scenarios['relevant_scenarios_bs']
    
           for num_reg in range(len(relevant_scenarios_bs)):
    
               nr                       = relevant_scenarios_bs[num_reg]
               relevant_idx_nr          = []
               relevant_idx_nr_max_zero = [] #Nr of idx larger than 0
               Intensity_measure_all    = np.zeros((probability_scenarios['nr_bs_scenarios'], len(POIs['name'])))
               scene_matrix             = np.transpose(Scenarios_BS[nr])
               convolved_scenarios_bs   = vec.dot(scene_matrix)
               relevant_file_scenario   = int(nr)-1
               isel                     = np.where(probability_scenarios['par_scenarios_bs'][:,0]==nr)
               arra                     = probability_scenarios['par_scenarios_bs'][isel]
               par_matrix               = np.transpose(arra[:,1:8])
               convolved_par_bs         = vec.dot(par_matrix)
               Iloc,nr_scenarios        = ismember(convolved_par_bs,convolved_scenarios_bs)
               idx_true_scenarios       = np.where([nr_scenarios>0])[1]
               probsel                  = probability_scenarios['ProbScenBS'][isel[0]]
               
               #infile = hazard_curves_files['gl_bs'][relevant_file_scenario]
               #print("       ---> Beg of Scenario: ", infile)
               #with h5py.File(infile, "r") as f:
               #    a_group_key = list(f.keys())[0]
               #    data = np.array(f.get(a_group_key))
    
               #IIM = data[str(relevant_file_scenario)][:, nr_scenarios]
               #print(relevant_file_scenario,hazard_curves_files)
               IIM = hazard_curves_files[str(relevant_file_scenario)][:, nr_scenarios]
               iselc=isel[0]
               for id_tmp in range(len(iselc)):
                  id_sel=iselc[id_tmp]
                  int_meas_bs[:,id_sel]=IIM[:,id_tmp]
    
           int_meas = int_meas_bs
    
        if len(probability_scenarios['ProbScenPS'])>1:
           if int(probability_scenarios['relevant_scenarios_ps'][0])>0:
              int_meas_ps              = np.zeros((len(selection),len(probability_scenarios['ProbScenPS'])))
              vec = np.array([100000000,100000,100,1,0.0100])
              relevant_scenarios_ps    = probability_scenarios['relevant_scenarios_ps']
              for num_reg in range(len(relevant_scenarios_ps)):
    
                  nr                       = int(relevant_scenarios_ps[num_reg])
                  relevant_idx_nr          = []
                  relevant_idx_nr_max_zero = [] #Nr of idx larger than 0
                  Intensity_measure_all    = np.zeros((probability_scenarios['nr_ps_scenarios'], len(POIs['name'])))
                  scene_matrix           = np.array(Scenarios_PS[nr]['Parameters'])
                  scene_matrix[:,3]      = np.array(Scenarios_PS[nr]['modelVal'][:])
                  scene_matrix[:,4]      = np.transpose(np.array(Scenarios_PS[nr]['SlipDistribution']))
                  scene_matrix           = np.transpose(scene_matrix)
                  convolved_scenarios_ps   = vec.dot(scene_matrix)
                  relevant_file_scenario   = int(nr)-1
                  isel                     = np.where(probability_scenarios['par_scenarios_ps'][:,0]==nr)
                  arra                     = probability_scenarios['par_scenarios_ps'][isel]
                  par_matrix               = np.transpose(arra[:,1:6])
                  convolved_par_ps         = vec.dot(par_matrix)
                  Iloc,nr_scenarios        = ismember(convolved_par_ps,convolved_scenarios_ps)
                  idx_true_scenarios       = np.where([nr_scenarios>0])[1]
                  probsel                  = probability_scenarios['ProbScenPS'][isel[0]]
                  
                  #infile = hazard_curves_files['gl_ps'][relevant_file_scenario]
                  #IIM = hazard_curves_files[str(relevant_file_scenario)][:, nr_scenarios]
                  #iselc=isel[0]
                  #print("       ---> Beg of Scenario: ", infile)
                  #with h5py.File(infile, "r") as f:
                  #    a_group_key = list(f.keys())[0]
                  #    data = np.array(f.get(a_group_key))
    
                  IIM = hazard_curves_files[str(relevant_file_scenario)][:, nr_scenarios]
                  iselc=isel[0]
                  for id_tmp in range(len(iselc)):
                     id_sel=iselc[id_tmp]
                     int_meas_ps[:,id_sel]=IIM[:,id_tmp]
              
              int_meas = int_meas_ps

        if len(probability_scenarios['ProbScenPS'])>1 and len(probability_scenarios['ProbScenBS'])>1 and int(probability_scenarios['relevant_scenarios_ps'][0])>0:
           int_meas = np.concatenate((int_meas_bs, int_meas_ps), axis=1)
     
        ### Selection of the POIs and Hobs/Hmod comparison ###
    
        # Selection of the closest POI to each observation location
        nbp=1
        tg_poi = {}
        tg_poi['lat'] = np.zeros((len(tg),nbp))
        tg_poi['lon'] = np.zeros((len(tg),nbp))
        tg_poi['id'] = np.zeros((len(tg),nbp))
        tg_poi['dist'] = np.zeros((len(tg),nbp))
        #tg_poi['wei'] = np.zeros((len(tg),nbp))
        tg_poi['ip'] = np.zeros((len(tg),nbp))
        name=POIs['name']
        selection=POIs['selected_pois']
        print('selection',len(selection))
        for tg_id in range(len(tg)):
            tglon=tg[tg_id,0]
            tglat=tg[tg_id,1]
            dist=np.ones((len(selection)))
            index=np.ones((len(selection)))
            for poi_id in range(len(selection)):
                idxp=name.index(selection[poi_id])
                lon=POIs['lon'][idxp]
                lat=POIs['lat'][idxp]
                dist[poi_id]=math.sqrt((tglon-lon)**2+(tglat-lat)**2)*111.0
                index[poi_id]=idxp
            for ip in range(nbp):
                idx = np.argmin(dist)
                idxp=name.index(selection[idx])
                tg_poi['id'][tg_id,ip] = idxp
                tg_poi['lon'][tg_id,ip] = POIs['lon'][idxp]
                tg_poi['lat'][tg_id,ip] = POIs['lat'][idxp]
                tg_poi['dist'][tg_id,ip] = dist[idx]
                tg_poi['ip'][tg_id,ip] = idx
                dist[idx]=100000
            #for ip in range(nbp):
            #    tmp = (1.0/tg_poi['dist'][tg_id,ip])/np.sum(1.0/tg_poi['dist'][tg_id])
            #    tg_poi['wei'][tg_id,ip] = tmp # tg_poi['dist'][tg_id,i]/np.sum(tg_poi['dist'][tg_id])
    
        # For each POI: selection of the maximal observation 
        tg_id=0
        while tg_id < len(tg):
            arr = tg_poi['id'][:,0]
            idxp = tg_poi['id'][tg_id,0]
            count = np.count_nonzero(arr == idxp)
            if count>1:
               where_arr = np.where(arr == idxp)
               position=where_arr[0]
               valu_max = np.argmax(tg[position,2])
               position = np.delete(position,valu_max)
               tg = np.delete(tg,position,axis=0)
               tg_poi['id'] = np.delete(tg_poi['id'],position,axis=0)
               tg_poi['ip'] = np.delete(tg_poi['ip'],position,axis=0)
               tg_poi['lon'] = np.delete(tg_poi['lon'],position,axis=0)
               tg_poi['lat'] = np.delete(tg_poi['lat'],position,axis=0)
               tg_poi['dist'] = np.delete(tg_poi['dist'],position,axis=0)
               #tg_poi['wei'] = np.delete(tg_poi['wei'],position,axis=0)
            tg_id += 1
    
        # Dividing the tide-gage and run-up value by 2
        # Real eventeal
        origin = tg[:,2]/2.0
        # Synth vent
        #origin = tg[:,2]
        omean=np.mean(origin)
    
        # Initialisation of the obs-mod comparison dict
        #probability_scenarios['all_ptf_val']=np.zeros((len(int_meas[0]),len((int_meas))))
        probability_scenarios['ptf_val']=np.zeros((len(int_meas[0]),len(tg)))
        probability_scenarios['obs_val']=np.zeros((len(int_meas[0]),len(tg)))
        probability_scenarios['ptf_obs_diff']=np.zeros((len(int_meas[0]),len(tg)))
        probability_scenarios['ptf_obs_norm']=np.zeros((len(int_meas[0]),len(tg)))
        probability_scenarios['NRMS']=np.zeros((len(int_meas[0])))
        probability_scenarios['tg_pois']=tg_poi
        probability_scenarios['tg_real']=tg
        n=float(len(tg))
        sigmasimu=1.0
    
        for i in range(len(int_meas[0])): #scenario
            for j in range(len(tg)): #pois
                datj=0.0
                jp=int(tg_poi['ip'][j,0])
                datj=int_meas[jp,i]
                if (datj<0.001):
                   datj=0.001
                # Real event
                datj=np.exp(np.log(datj)+(sigmasimu**2)/2.0)
                # Sytnth event
                # datj=datj
                orig=origin[j]
                if (datj<0.001):
                    datj=0.0
                if (orig<0.001):
                    orig=0.0
                probability_scenarios['ptf_val'][i,j]=datj
                probability_scenarios['obs_val'][i,j]=orig
                probability_scenarios['ptf_obs_diff'][i,j]=datj-orig
                if orig>0.001:
                   probability_scenarios['ptf_obs_norm'][i,j]=(datj-orig)/orig
                else:
                   probability_scenarios['ptf_obs_norm'][i,j]=-9999
    
        if name_type_ens == 'ori_ensemble':
           ptf_out['probability_scenarios']['ptf_val']=probability_scenarios['ptf_val']
           ptf_out['probability_scenarios']['obs_val']=probability_scenarios['obs_val']
           print('Has entered tutti step')
        else:
           ptf_out[name_type_ens][Nid]=probability_scenarios
    

    return ptf_out

