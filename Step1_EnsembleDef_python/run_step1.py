#!/usr/bin/env python
#!/home/louise/miniconda3/bin/python3.8

# Import system modules
import os
import ast
import sys
import utm
import threading
import configparser
import ray
import h5py
import hickle as hkl
import json
import pprint
import math
import numpy       as np
import copy
from time     import gmtime
from time     import strftime
from datetime import datetime

# adding the path to find some modules
#sys.path.append('Step1_EnsembleDef_python/py')
sys.path.append('../Commons/py')
# Import functions from pyPTF modules
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
from ptf_save                  import save_ptf_out_up
from ptf_ellipsoids          import build_location_ellipsoid_objects
from ptf_ellipsoids          import build_ellipsoid_objects
from ptf_mix_utilities        import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_short_term                 import short_term_probability_distribution
from ptf_probability_scenarios      import compute_probability_scenarios
from ptf_ensemble_sampling_MC       import compute_ensemble_sampling_MC
from ptf_ensemble_sampling_RS       import compute_ensemble_sampling_RS
from ptf_ensemble_kagan             import compute_ensemble_kagan
from ptf_ensemble_mare             import compute_ensemble_mare
from ptf_ensemble_hmax             import extract_hmax
from ptf_ensemble_resampling       import mc_samp
from ptf_hazard_curves         import compute_hazard_curves
from ptf_preload_curves        import load_hazard_values
#from ptf_hazard_curves         import compute_hazard_curves
#from ptf_alert_levels          import set_alert_levels

#from ptf_messages            import create_message_matrix
# from ttt                     import run_ttt
# from ttt_map_utilities       import extract_contour_times
# from ttt_map_utilities       import build_tsunami_travel_time_map

def step1_ensembleEval(**kwargs):

    Scenarios_PS     = kwargs.get('Scenarios_PS', None)
    Scenarios_BS     = kwargs.get('Scenarios_BS', None)
    LongTermInfo     = kwargs.get('LongTermInfo', None)
    POIs             = kwargs.get('POIs', None)
    PSBarInfo        = kwargs.get('PSBarInfo', None)
    Mesh             = kwargs.get('Mesh', None)
    Region_files     = kwargs.get('Region_files', None)
    args             = kwargs.get('args', None)
    Config           = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_data', None)

    OR_EM=int(Config.get('Sampling','OR_EM'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    RS_output_scen=int(Config.get('Sampling','RS_output_scen'))
    Kagan_weights=int(Config.get('Sampling','Kagan_weights'))
    NbrFM=int(Config.get('Sampling','NbrFM'))
    Mare_weights=int(Config.get('Sampling','Mare_weights'))
    FM_path=Config.get('EventID','FM_path')
    TSU_path=Config.get('EventID','TSU_path')
    EventID=Config.get('EventID','eventID')
    samp = int(Config.get('Sampling','New_samp'))
    up_time=ast.literal_eval(Config.get('Sampling','up_time'))
    ptf_out = dict()

    ### Loading and selection of the focal mechanism ###
    h5file = FM_path
    h5file = '../IO/med-tsumaps-python/tsunami_data_all.hdf5'
    FM_file=hkl.load(h5file)
    totfm = len(FM_file[EventID])
    if NbrFM==0 or NbrFM>=totfm:
       focal_mech=np.array([FM_file[EventID]])
    else:
       focal_mech=FM_file[EventID][0:NbrFM,:]
       focal_mech=np.array([focal_mech])
    Nfm=len(focal_mech)

    ### Loading of the tsunami data ###
    h5file = TSU_path
    h5file = '../IO/med-tsumaps-python/tsunami_data_all.hdf5'
    TSU_file=hkl.load(h5file)
    tsu_data=TSU_file[EventID]
    Ntsu=len(tsu_data)
    half_tsu=math.ceil(len(tsu_data)/2)
    if EventID=='2017_0720_kos-bodrum':
       half_tsu=3
    elif EventID=='2020_1030_samos':
       half_tsu=5
    tsu_data_wei=tsu_data[0:half_tsu]
    tsu_data_test=tsu_data[half_tsu:-1]

    print('############## Initialization of the ensemble manager #################')

    print('Build ellipsoids objects')
    ellipses = build_ellipsoid_objects(event = event_parameters,
                                       cfg   = Config,
                                       args  = args)


    print('Conversion to utm')
    LongTermInfo, POIs, PSBarInfo = conversion_to_utm(longTerm  = LongTermInfo,
                                                      Poi       = POIs,
                                                      event     = event_parameters,
                                                      PSBarInfo = PSBarInfo)

    ##########################################################
    # Set separation of lambda BS-PS
    print('Separation of lambda BS-PS')
    lambda_bsps = load_lambda_BSPS(cfg                   = Config,
                                   args                  = args,
                                   event_parameters      = event_parameters,
                                   LongTermInfo          = LongTermInfo)


    lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                         args             = args,
                                         event_parameters = event_parameters,
                                         lambda_bsps      = lambda_bsps,
                                         LongTermInfo     = LongTermInfo,
                                         mesh             = Mesh)

    #print(lambda_bsps['regionsPerPS'])
    #sys.exit()
    ##########################################################
    # Pre-selection of the scenarios
    #
    # Magnitude: First PS then BS
    # At this moment the best solution is to insert everything into a dictionary (in matlab is the PreSelection structure)
    print('Pre-selection of the Scenarios')
    pre_selection = pre_selection_of_scenarios(cfg                = Config,
                                               args               = args,
                                               event_parameters   = event_parameters,
                                               LongTermInfo       = LongTermInfo,
                                               PSBarInfo          = PSBarInfo,
                                               ellipses           = ellipses)

    if(pre_selection == False):
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        status             = 'end')
        return False

    
    if OR_EM>0:
       
        print('############## Initial ensemble #################')
 
        ##########################################################
        # COMPUTE PROB DISTR
        #
        #    Equivalent of shortterm.py with output: node_st_probabilities
        #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
        print('Compute short term probability distribution')

        short_term_probability  = short_term_probability_distribution(cfg                = Config,
                                                                      args               = args,
                                                                      event_parameters   = event_parameters,
                                                                      LongTermInfo       = LongTermInfo,
                                                                      PSBarInfo          = PSBarInfo,
                                                                      lambda_bsps        = lambda_bsps,
                                                                      pre_selection      = pre_selection)

        if(short_term_probability == True):
            ptf_out = save_ptf_dictionaries(cfg                = Config,
                                            args               = args,
                                            event_parameters   = event_parameters,
                                            status             = 'end')
            return False

        ##COMPUTE PROBABILITIES SCENARIOS: line 840
        print('Compute Probabilities scenarios')
        probability_scenarios = compute_probability_scenarios(cfg                = Config,
                                                              args               = args,
                                                              event_parameters   = event_parameters,
                                                              LongTermInfo       = LongTermInfo,
                                                              PSBarInfo          = PSBarInfo,
                                                              lambda_bsps        = lambda_bsps,
                                                              pre_selection      = pre_selection,
                                                              regions            = Region_files,
                                                              short_term         = short_term_probability,
                                                              Scenarios_PS       = Scenarios_PS)

    ptf_out['probability_scenarios']  = probability_scenarios
    ptf_out['POIs']                   = POIs

    if samp>2:
       type_ens = 'ori'
       name_type_ens='ori_ensemble'
       #ptf_out = mc_samp(ptf_out,name_type_ens,type_ens,MC_samp_scen)
       ptf_out = extract_hmax(cfg                = Config,
                              event_parameters   = event_parameters,
                              args               = args,
                              ptf_out            = ptf_out,
                              LongTermInfo       = LongTermInfo,
                              pois               = POIs,
                              h_curve_files      = hazard_curves_files,
                              type_ens           = type_ens,
                              name_type_ens      = name_type_ens,
                              tsu_data           = tsu_data)


################### Discretized LH or MC sampling ########################
    
    if MC_samp_scen>0: 
       print('############## Monte Carlo sampling #################')
       sampled_ensemble_MC = compute_ensemble_sampling_MC(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)
       ptf_out['new_ensemble_MC']           = sampled_ensemble_MC
      
       data=dict()
       relevant_scenarios_bs    = probability_scenarios['relevant_scenarios_bs']
       for num_reg in range(len(relevant_scenarios_bs)):
           nr                       = relevant_scenarios_bs[num_reg]
           relevant_file_scenario   = int(nr)-1
           infile = hazard_curves_files['gl_bs'][relevant_file_scenario]
           print("       ---> Beg of Scenario: ", infile)
           print("       ---> Region : ", str(relevant_file_scenario))
           with h5py.File(infile, "r") as f:
               a_group_key = list(f.keys())[0]
               data[str(relevant_file_scenario)] = {}
               data[str(relevant_file_scenario)] = np.array(f.get(a_group_key))
 
       
       type_ens='MC' 
       if Kagan_weights>0:

          for val in up_time:

              if val>0:
                 ptf_out_kag = copy.deepcopy(ptf_out)
                 ptf_out_kag = compute_ensemble_kagan(cfg                = Config,
                                                  args               = args,
                                                  event_parameters   = event_parameters,
                                                  ptf_out            = ptf_out_kag,
                                                  focal_mechanism    = focal_mech,
                                                  type_ens           = type_ens)
                 ptf_out_kag['new_ensemble_MC_hmax'] = ptf_out_kag['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_kag = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_kag,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data)
              else:
                 ptf_out_kag = copy.deepcopy(ptf_out)
                 ptf_out_kag['new_ensemble_MC_hmax'] = ptf_out_kag['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_kag = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_kag,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data)

              if samp>0:
                 name_type_ens='samp_ensemble_MC'
                 ptf_out_kag = mc_samp(ptf_out_kag,name_type_ens,type_ens,MC_samp_scen)
                 ptf_out_kag = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_kag,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data)

              if val>0:
                 # Save outputs
                 print("Save pyPTF output")
                 val='kag'
                 saved_files = save_ptf_out_up(cfg                = Config,
                                                    args               = args,
                                                    event_parameters   = event_parameters,
                                                    ptf                = ptf_out_kag,
                                                    up_time            = val
                                                    )
              else:
                 # Save outputs
                 print("Save pyPTF output")
                 val='nokag'
                 saved_files = save_ptf_out_up(cfg                = Config,
                                                    args               = args,
                                                    event_parameters   = event_parameters,
                                                    ptf                = ptf_out_kag,
                                                    up_time            = val
                                                    )


       if Mare_weights>0:

          for val in up_time:

              if val>0:
                 ptf_out_up = copy.deepcopy(ptf_out)
                 name_type_ens = 'new_ensemble_MC'
                 ptf_out_up = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_up,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_wei)
                 ptf_out_up = compute_ensemble_mare(cfg                = Config,
                                                 event_parameters   = event_parameters,
                                                 args               = args,
                                                 ptf_out            = ptf_out_up,
                                                 LongTermInfo       = LongTermInfo,
                                                 pois               = POIs,
                                                 h_curve_files      = hazard_curves_files,
                                                 type_ens           = type_ens,
                                                 tsu_data           = tsu_data_wei)
                 ptf_out_up['new_ensemble_MC_hmax'] = ptf_out_up['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_up = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_up,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              else:
                 ptf_out_up = copy.deepcopy(ptf_out)
                 ptf_out_up['new_ensemble_MC_hmax'] = ptf_out_up['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_up = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_up,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              if samp>0:
                 name_type_ens='samp_ensemble_MC'
                 ptf_out_up = mc_samp(ptf_out_up,name_type_ens,type_ens,MC_samp_scen)
                 ptf_out_up = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_up,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              if val>0:
                   # Save outputs
                   print("Save pyPTF output")
                   val='tsu'
                   saved_files = save_ptf_out_up(cfg                = Config,
                                                      args               = args,
                                                      event_parameters   = event_parameters,
                                                      ptf                = ptf_out_up,
                                                      up_time            = val
                                                      )
              else:
                   # Save outputs
                   print("Save pyPTF output")
                   val='noweight'
                   saved_files = save_ptf_out_up(cfg                = Config,
                                                         args               = args,
                                                         event_parameters   = event_parameters,
                                                         ptf                = ptf_out_up,
                                                         up_time            = val)

       if Kagan_weights>0 and Mare_weights>0:

          for val in up_time:

              if val>0:
                 ptf_out_km = copy.deepcopy(ptf_out)
                 name_type_ens = 'new_ensemble_MC'
                 ptf_out_km = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_km,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_wei)

                 ptf_out_km = compute_ensemble_kagan(cfg                = Config,
                                                  args               = args,
                                                  event_parameters   = event_parameters,
                                                  ptf_out            = ptf_out_km,
                                                  focal_mechanism    = focal_mech,
                                                  type_ens           = type_ens)

                 ptf_out_km = compute_ensemble_mare(cfg                = Config,
                                                 event_parameters   = event_parameters,
                                                 args               = args,
                                                 ptf_out            = ptf_out_km,
                                                 LongTermInfo       = LongTermInfo,
                                                 pois               = POIs,
                                                 h_curve_files      = hazard_curves_files,
                                                 type_ens           = type_ens,
                                                 tsu_data           = tsu_data_wei)

                 ptf_out_km['new_ensemble_MC_hmax'] = ptf_out_km['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_km = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_km,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              else:
                 ptf_out_km = copy.deepcopy(ptf_out)
                 ptf_out_km['new_ensemble_MC_hmax'] = ptf_out_km['new_ensemble_MC']
                 name_type_ens = 'new_ensemble_MC_hmax'
                 ptf_out_km = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_km,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              if samp>0:
                 name_type_ens='samp_ensemble_MC'
                 ptf_out_km = mc_samp(ptf_out_km,name_type_ens,type_ens,MC_samp_scen)
                 ptf_out_km = extract_hmax(cfg                = Config,
                                        event_parameters   = event_parameters,
                                        args               = args,
                                        ptf_out            = ptf_out_km,
                                        LongTermInfo       = LongTermInfo,
                                        pois               = POIs,
                                        #h_curve_files      = hazard_curves_files,
                                        h_curve_files      = data,
                                        type_ens           = type_ens,
                                        name_type_ens      = name_type_ens,
                                        tsu_data           = tsu_data_test)

              if val>0:
                   # Save outputs
                   print("Save pyPTF output")
                   val='tsukag'
                   saved_files = save_ptf_out_up(cfg                = Config,
                                                         args               = args,
                                                         event_parameters   = event_parameters,
                                                         ptf                = ptf_out_km,
                                                         up_time            = val)
              else:
                   # Save outputs
                   print("Save pyPTF output")
                   val='noweight'
                   saved_files = save_ptf_out_up(cfg                = Config,
                                                         args               = args,
                                                         event_parameters   = event_parameters,
                                                         ptf                = ptf_out_km,
                                                         up_time            = val)



       #if MC_output_scen>0:
       #   for Nid in range(MC_samp_run):
       #       MC_samp_scen=len(ptf_out['new_ensemble_MC'][Nid]['par_scenarios_bs'][:,0])
       #       par=np.zeros((11))
       #       myfile = open('./ptf_localOutput/list_mc_nb%d_of_%d_scenbs.txt'%(Nid,MC_samp_scen), 'w')
       #       for Nscen in range(MC_samp_scen):
       #           #for ipar in range(11):
       #           par[:]=ptf_out['new_ensemble_MC'][Nid]['par_scenarios_bs'][Nscen,:]
       #           myfile.write("%f %f %f %f %f %f %f %f %f %f\n"%(par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8],par[9],par[10]))
       #       myfile.close()


########### Updtae of the ptf_out ###########

    if(probability_scenarios == False):
        print( "--> No Probability scenarios found. Save and Exit")
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        status             = 'end')
        return False

    # in order to plot here add nested dict to ptf_out
    ptf_out['short_term_probability'] = short_term_probability
    ptf_out['event_parameters']       = event_parameters
    ptf_out['POIs']                   = POIs


    print('End pyPTF')

    return ptf_out



################################################################################################
#                                                                                              #
#                                  BEGIN                                                       #
################################################################################################

############################################################
# Read Stdin
#print('\n')
args=parse_ptf_stdin()

############################################################
# Initialize and load configuaration file
cfg_file        = args.cfg
Config          = configparser.RawConfigParser()
Config.read(cfg_file)
Config          = update_cfg(cfg=Config, args=args)
min_mag_message = float(Config.get('matrix','min_mag_for_message'))

############################################################
#LOAD INFO FROM SPTHA
PSBarInfo                                         = load_PSBarInfo(cfg=Config, args=args)
hazard_curves_files                               = load_hazard_values(cfg=Config, args=args, in_memory=True)
Scenarios_PS, Scenarios_BS                        = load_Scenarios_Reg(cfg=Config, args=args, in_memory=True)
LongTermInfo, POIs, Mesh, Region_files            = ptf_preload(cfg=Config, args=args)

begin_of_time = datetime.utcnow()
end_of_time = datetime.utcnow()
diff_time        = end_of_time - begin_of_time
print("--> Execution Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds))
#sys.exit()

begin_of_time = datetime.utcnow()

#### Load event parameters then workflow and ttt are parallel
#print('############################')
print('Load event parameters')
# Load the event parameters from json file consumed from rabbit
event_parameters = load_event_parameters(event       = args.event,
                                         format      = args.event_format,
                                         routing_key = 'INT.QUAKE.CAT',
                                         args        = args,
                                         json_rabbit = None,
                                         cfg         = Config)
print_event_parameters(dict=event_parameters, args = args)

# --------------------------------------------------- #
# check  inneam, mag_action are true or false
# Qui controlla se la magnitudo minima indicata dalla matrice decisionale e' raggiunta
# e se l'evento si trova nell'area neam. nel primo caso, se la magnitudo e' sotto soglia,
# la procedura invia un messaggio al rabbit contentente l'informazione 'magnitude < 5.5, nothing to do'
# che verra mostrata su jet.
# La differenza tra area neam o fuori, per ora e' solo legata alla differenza dei forecast point da caricare
# Le differenze di invio sono gestite da catcom@tigerX
# --------------------------------------------------- #
#event_parameters = check_if_neam_event(dictionary=event_parameters, cfg=Config)
#print(" --> Event INNEAM:            ", event_parameters['inneam'])

# --------------------------------------------------- #
# check if event inland with respect neam guidelines
# event_parameters, geometry_land_with_point         = chck_if_point_is_land(event_parameters=event_parameters, cfg=Config)
# event_parameters['epicentral_distance_from_coast'] = get_distance_point_to_Ring(land_geometry=geometry_land_with_point, event_parameters=event_parameters)
ptf_out = save_ptf_dictionaries(cfg                = Config,
                                args               = args,
                                event_parameters   = event_parameters,
                                status             = 'new')

#print(" --> Epicenter INLAND:           %r  (%.3f [km]) " % (event_parameters['epicenter_is_in_land'],  event_parameters['epicentral_distance_from_coast'] ))
#nr_cpu_allowed        = float(Config.get('Settings','nr_cpu_max'))
#print(" --> Initialize ray with %d cpu" % (int(nr_cpu_allowed)))
#ray.init(num_cpus=int(nr_cpu_allowed), include_dashboard=False, ignore_reinit_error=True)
#print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    
######################################################
# Ensemble evaluation

ptf_out = step1_ensembleEval(Scenarios_PS = Scenarios_PS,
                                        Scenarios_BS = Scenarios_BS,
                                        LongTermInfo = LongTermInfo,
                                        POIs         = POIs,
                                        PSBarInfo    = PSBarInfo,
                                        Mesh         = Mesh,
                                        Region_files = Region_files,
                                        #h_curve_files= hazard_curves_files,
                                        args         = args,
                                        cfg          = Config,
                                        event_data   = event_parameters)


######################################################
# Save outputs
print("Save pyPTF output")
saved_files = save_ptf_out(cfg                = Config,
                           args               = args,
                           event_parameters   = event_parameters,
                           ptf                = ptf_out,
                           #status             = status,
                           )

#saved_files = save_ptf_dictionaries(cfg                = Config,
#                                        args               = args,
#                                        event_parameters   = event_parameters,
#                                        ptf                = ptf_out,
#                                        #status             = status,
#                                        )
#
#
#######################################################
## Make figures from dictionaries
#print("Make pyPTF figures")
#saved_files = make_ptf_figures(cfg                = Config,
#                                   args               = args,
#                                   event_parameters   = event_parameters,
#                                   ptf                = ptf_out,
#                                   saved_files        = saved_files)
#
#print("Save some extra usefull txt values")
saved_files = save_ptf_as_txt(cfg                = Config,
                                  args               = args,
                                  event_parameters   = event_parameters,
                                  ptf                = ptf_out,
                                  #status             = status,
                                  pois               = ptf_out['POIs'],
                                  #alert_levels       = ptf_out['alert_levels'],
                                  saved_files        = saved_files,
                                  #fcp                = fcp_merged,
                                  ensembleYN         = True
                                  )

#
#
