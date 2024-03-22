import os
import sys
import ast
import ray
import numpy as np
import configparser
import hickle as hkl
import h5py
sys.path.append('../Commons/py')
from ptf_preload               import ptf_preload
from ptf_preload               import load_Scenarios_Reg
from ptf_preload             import load_PSBarInfo
#from ptf_preload_curves      import load_hazard_values
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import parse_ptf_stdin
from ptf_parser              import update_cfg
from ptf_parser                import parse_ptf_stdin
from ptf_parser                import update_cfg
from ptf_mix_utilities         import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_ellipsoids            import build_location_ellipsoid_objects
from ptf_ellipsoids            import build_ellipsoid_objects
from ptf_short_term            import short_term_probability_distribution
from ptf_probability_scenarios import compute_probability_scenarios
from ptf_hazard_curves         import compute_hazard_curves
from ptf_preload_curves        import load_hazard_values
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import load_ptf_out
from ptf_save                  import save_ptf_out
from ptf_figures               import make_ptf_figures
from ptf_save                  import save_ptf_as_txt
from datetime import datetime

def step3_hazard(**kwargs):

    Config                  = kwargs.get('Config', None)
    args                    = kwargs.get('args', None)
    POIs                    = kwargs.get('POIs', None)
    event_parameters        = kwargs.get('event_parameters', None)
    Scenarios_BS            = kwargs.get('Scenarios_BS', None)
    Scenarios_PS            = kwargs.get('Scenarios_PS', None)
    LongTermInfo            = kwargs.get('LongTermInfo', None)
    h_curve_files           = kwargs.get('h_curve_files', None)
    ptf_out                 = kwargs.get('ptf_out', None)

    OR_HC=int(Config.get('Sampling','OR_HC'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    short_term_probability = ptf_out['short_term_probability']
    probability_scenarios = ptf_out['probability_scenarios']
    up_time=ast.literal_eval(Config.get('Sampling','up_time'))
    Kagan_weights=int(Config.get('Sampling','Kagan_weights'))
    Mare_weights=int(Config.get('Sampling','Mare_weights'))


    ######################################################
    # Compute hazard curves
    
    if OR_HC>0:
       print('Compute hazard curves at POIs')
       begin_of_utcc       = datetime.utcnow()
       hazard_curves = compute_hazard_curves(cfg                = Config,
                                             args               = args,
                                             pois               = POIs,
                                             event_parameters   = event_parameters,
                                             probability_scenarios  = probability_scenarios,
                                             Scenarios_BS       = Scenarios_BS,
                                             Scenarios_PS       = Scenarios_PS,
                                             LongTermInfo       = LongTermInfo,
                                             h_curve_files      = h_curve_files,
                                             short_term_probability = short_term_probability)
       end_of_utcc       = datetime.utcnow()
       diff_time        = end_of_utcc - begin_of_utcc
       print(" --> Building Hazard curves in Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds/1000))
       ######################################################      
       # in order to plot here add nested dict to ptf_out
       ptf_out['hazard_curves_original']          = hazard_curves

    ######################################################
    # Compute hazard curves

    if MC_samp_scen>0: 

       rangenid=len(ptf_out['new_ensemble_MC'])
       if Mare_weights>0 and Kagan_weights>0:
          valarr=['nokag','noweight','kag','tsu','tsukag']
       elif Mare_weights>0:
          valarr=['noweight','tsu']
       elif Kagan_weights>0:
          valarr=['nokag','kag']
       else:
          valarr=['noweight']

       for Nid in range(rangenid):

           #for val in up_time:
           for ival in range(len(valarr)):
   
               val=valarr[ival]
               print('HC : ',val)       
               ptf_out={}
               h5file = './ptf_localOutput/ptf_out_t' +str(val)+ '.hdf5'
               if Mare_weights<1 and Kagan_weights<1:
                  h5file = './ptf_localOutput/ptf_out.hdf5'
               ptf_out=hkl.load(h5file)
               probability_scenarios = ptf_out['new_ensemble_MC'][Nid]
               hazard_curves = compute_hazard_curves(cfg                = Config,
                                                     args               = args,
                                                     pois               = POIs,
                                                     event_parameters   = event_parameters,
                                                     probability_scenarios  = probability_scenarios,
                                                     Scenarios_BS       = Scenarios_BS,
                                                     Scenarios_PS       = Scenarios_PS,
                                                     LongTermInfo       = LongTermInfo,
                                                     h_curve_files      = h_curve_files,
                                                     short_term_probability = short_term_probability)
               ######################################################
    
               # in order to plot here add nested dict to ptf_out
    
               ptf_out['hazard_curves_MC_%d'%Nid]          = hazard_curves
    
               print('HC done : ',val)
               h5file = './ptf_localOutput/hazard_curves_t' +str(val)+ '.hdf5'
               hf = h5py.File(h5file, 'w')
               hf.create_dataset('hazard_curves_at_pois', data=hazard_curves['hazard_curves_at_pois'])
               hf.create_dataset('hazard_curves_at_pois_mean', data=hazard_curves['hazard_curves_at_pois_mean'])
               #if (ptf_out['new_ensemble_MC']['nr_bs_scenarios'] > 0):
               #    hf.create_dataset('hazard_curves_bs_at_pois', data=hazard_curves['bs']['hazard_curves_bs_at_pois'])
               #    hf.create_dataset('hazard_curves_bs_at_pois_mean', data=hazard_curves['bs']['hazard_curves_bs_at_pois_mean'])
               #    hf.create_dataset('tsunami_intensity_name', data=hazard_curves['tsunami_intensity_name'])
               #    hf.create_dataset('Intensity_measure_all_bs', data=hazard_curves['Intensity_measure_all_bs'])
               #if (ptf_out['new_ensemble_MC']['nr_ps_scenarios'] > 0):
               #    hf.create_dataset('hazard_curves_ps_at_pois', data=hazard_curves['ps']['hazard_curves_ps_at_pois'])
               #    hf.create_dataset('hazard_curves_ps_at_pois_mean', data=hazard_curves['ps']['hazard_curves_ps_at_pois_mean'])
               #else:
               #    hf.create_dataset('hazard_curves_ps_at_pois', data=np.array([0.0]))
               #    hf.create_dataset('hazard_curves_ps_at_pois_mean', data=np.array([0.0]))
               hf.close()
               print('HC saved : ',val)

    ######################################################
    # Compute hazard curves

    if RS_samp_scen>0:
       rangenid=len(ptf_out['new_ensemble_RS'])
       for Nid in range(rangenid):
           probability_scenarios = ptf_out['new_ensemble_RS'][Nid]
           print('Compute hazard curves at POIs')
           begin_of_utcc       = datetime.utcnow()
           hazard_curves = compute_hazard_curves(cfg                = Config,
                                                 args               = args,
                                                 pois               = POIs,
                                                 event_parameters   = event_parameters,
                                                 probability_scenarios  = probability_scenarios,
                                                 Scenarios_BS       = Scenarios_BS,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 LongTermInfo       = LongTermInfo,
                                                 h_curve_files      = h_curve_files,
                                                 short_term_probability = short_term_probability)
           end_of_utcc       = datetime.utcnow()
           diff_time        = end_of_utcc - begin_of_utcc
           print(" --> Building Hazard curves in Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds/1000))
           ######################################################

           # in order to plot here add nested dict to ptf_out

           ptf_out['hazard_curves_RS_%d'%Nid]          = hazard_curves

    return ptf_out

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
h_curve_files                                     = load_hazard_values(cfg=Config, args=args, in_memory=True)
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
ptf_out = load_ptf_out(cfg=Config, args=args, event_parameters=event_parameters)  #status= status,)

ptf_out = step3_hazard(Config                   = Config,
                       args                     = args,
                       POIs                     = POIs,
                       event_parameters         = event_parameters,
                       Scenarios_BS             = Scenarios_BS,
                       Scenarios_PS             = Scenarios_PS,
                       LongTermInfo             = LongTermInfo,
                       h_curve_files            = h_curve_files,
		       ptf_out                  = ptf_out)


######################################################
## Save outputs
#print("Save pyPTF output")
#saved_files = save_ptf_out(cfg                = Config,
#                           args               = args,
#                           event_parameters   = event_parameters,
#                           ptf                = ptf_out,
#                           #status             = status,
#                           )

saved_files = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        ptf                = ptf_out,
                                        #status             = status,
                                        )


######################################################
## Make figures from dictionaries
#print("Make pyPTF figures")
#saved_files = make_ptf_figures(cfg                = Config,
#                                   args               = args,
#                                   event_parameters   = event_parameters,
#                                   ptf                = ptf_out,
#                                   saved_files        = saved_files)
#
print("Save some extra usefull txt values")
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


