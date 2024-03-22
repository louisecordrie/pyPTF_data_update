import sys
import time
import numpy as np
from random import seed
from random import random
from numba import jit
from scipy import stats
import scipy
from numpy.linalg import inv
import emcee
import copy
import math
from math import radians, cos, sin, asin, sqrt

def compute_ensemble_sampling_MC(**kwargs):


    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    args           = kwargs.get('args', None)
    LongTermInfo   = kwargs.get('LongTermInfo', None)
    PSBarInfo      = kwargs.get('PSBarInfo', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    pre_selection  = kwargs.get('pre_selection', None)
    short_term     = kwargs.get('short_term', None)
    regions        = kwargs.get('regions', None)
    Scenarios_PS   = kwargs.get('Scenarios_PS', None)
    probability_scenarios  = kwargs.get('proba_scenarios', None)

    #print('Probability BS : ',probability_scenarios['ProbScenBS'])
    print('Probability PS : ',probability_scenarios['ProbScenPS'])

    if len(probability_scenarios['ProbScenBS'])>1 and len(probability_scenarios['ProbScenPS'])>1:
       TotProbBS_all = np.sum(probability_scenarios['ProbScenBS'])
       TotProbPS_all = np.sum(probability_scenarios['ProbScenPS'])
    elif len(probability_scenarios['ProbScenBS']):
       print('Only BS')
       TotProbBS_all = 1.0
       TotProbPS_all = 0.0
    else:
       print('Only PS')
       TotProbBS_all = 0.0
       TotProbPS_all = 1.0

    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    MC_type = Config.get('Sampling','MC_type')
    Kagan_weights=int(Config.get('Sampling','Kagan_weights'))
    Nsamp=np.ones(MC_samp_run)*MC_samp_scen
    sampled_ensemble = {}

    Nid=0
    ### Loop of Nsamp MC sampling ###
    for N in Nsamp:
        ### Begining of the creation of the Nth sampled ensemble ###
        N=int(N)
        NBS= int(TotProbBS_all*N) ### NBS: number of scenarios sampled from the BS ensemble
        NPS= N-NBS ### NPS: number of scenarios sampled from the PS ensemble
        print("Number of scenarios",NBS,NPS)
        # NBS and NPS are proportionnal to the probability of PS and BS
        sampled_ensemble[Nid] = dict()
        sampled_ensemble[Nid] = set_if_compute_scenarios(cfg           = Config,
                                                short_term    = short_term)
        prob_len_BS = len(probability_scenarios['ProbScenBS'])
        prob_len_PS = len(probability_scenarios['ProbScenPS'])

        ### Creation of the array of cumulated probability intervals associated to the initial ensemble ###
        intervals_ensemble_BS = np.zeros(prob_len_BS)
        prob_cum = 0
        for i in range(prob_len_BS):
            prob_cum=prob_cum+probability_scenarios['ProbScenBS'][i]
            intervals_ensemble_BS[i]= prob_cum

        ### Creation of the array of cumulated probability intervals associated to the initial ensemble ###
        intervals_ensemble_PS = np.zeros(prob_len_PS)
        prob_cum = 0
        for i in range(prob_len_PS):
            prob_cum=prob_cum+probability_scenarios['ProbScenPS'][i]
            intervals_ensemble_PS[i]= prob_cum

        ### Initialization of the dictionnaries ###
        sampled_ensemble[Nid]['prob_scenarios_bs_fact'] = np.zeros( (NBS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_bs'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['prob_scenarios_ang'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['par_scenarios_bs'] = np.zeros(  (NBS, 11) )
        sampled_ensemble[Nid]['prob_scenarios_ps_fact'] = np.zeros( (NPS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_ps'] = np.zeros( (NPS) )
        sampled_ensemble[Nid]['par_scenarios_ps'] = np.zeros(  (NPS,  7) )
        sampled_ensemble[Nid]['iscenbs']=np.zeros(NBS)
        sampled_ensemble[Nid]['iscenps']=np.zeros(NPS)

        sampled_ensemble = bs_probability_scenarios(cfg              = Config,
                                                     short_term       = short_term,
                                                     pre_selection    = pre_selection,
                                                     regions_files    = regions,
                                                     prob_scenes      = probability_scenarios,
                                                     samp_ens         = sampled_ensemble,
                                                     Discretizations  = LongTermInfo['Discretizations'],
						     NBS	      = NBS,
                                                     Nid	      = Nid,
                                                     intervals_ensemble = intervals_ensemble_BS)

        sampled_ensemble = ps_probability_scenarios(cfg              = Config,
                                                         PSBarInfo        = PSBarInfo,
                                                         short_term       = short_term,
                                                         pre_selection    = pre_selection,
                                                         prob_scenes      = probability_scenarios,
                                                         samp_ens         = sampled_ensemble,
                                                         region_ps        = LongTermInfo['region_listPs'],
                                                         Model_Weights    = LongTermInfo['Model_Weights'],
                                                         Scenarios_PS     = Scenarios_PS,
                                                         ps1_magnitude    = LongTermInfo['Discretizations']['PS-1_Magnitude'],
                                                         lambda_bsps      = lambda_bsps,
                                                         NPS              = NPS,
							 Nid              = Nid,
                                                         intervals_ensemble = intervals_ensemble_PS)
    
        if(sampled_ensemble[Nid] == False):
            return False

        # check on the nr scenarios computed into the two section. Should be identical
        check_bs = 'OK'
        check_ps = 'OK'

        if(sampled_ensemble[Nid]['nr_bs_scenarios'] != short_term['Total_BS_Scenarios']):
            check_bs = 'WARNING'
        if(sampled_ensemble[Nid]['nr_ps_scenarios'] != short_term['Total_PS_Scenarios']):
            check_ps = 'WARNING'

        print(' --> Check Nr Bs scenarios: %7d  <--> %7d --> %s' % (sampled_ensemble[Nid]['nr_bs_scenarios'], short_term['Total_BS_Scenarios'], check_bs))
        print(' --> Check Nr Ps scenarios: %7d  <--> %7d --> %s' % (sampled_ensemble[Nid]['nr_ps_scenarios'], short_term['Total_PS_Scenarios'], check_ps))
        
        # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
        ProbScenBS        = sampled_ensemble[Nid]['prob_scenarios_bs_fact'].prod(axis=1)
        ProbScenPS        = sampled_ensemble[Nid]['prob_scenarios_ps_fact'].prod(axis=1)
        #print(len(ProbScenBS))
        TotProbBS_preNorm = np.sum(ProbScenBS)
        TotProbPS_preNorm = np.sum(ProbScenPS)
        TotProb_preNorm   = TotProbBS_preNorm + TotProbPS_preNorm

        # No scenarios bs or ps possible
        if(TotProb_preNorm == 0):
            return False
        elif(TotProb_preNorm != 0):
            ProbScenBS = ProbScenBS / TotProb_preNorm
            ProbScenPS = ProbScenPS / TotProb_preNorm
            print(' --> Total Bs scenarios probability pre-renormalization: %.5f' % TotProbBS_preNorm)
            print(' --> Total Ps scenarios probability pre-renormalization: %.5f' % TotProbPS_preNorm)
            print('     --> Total Bs and Ps probabilty renormalized to 1')

        TotBS=len(ProbScenBS)
        TotPS=len(ProbScenPS)
        Tot=TotBS+TotPS
        ProbScenBS = np.ones(TotBS)
        ProbScenPS = np.ones(TotPS)

        ######### Re-initialisation of the probability ######
        ## A uniform probability is attributed to all the new scenarios ##
        ## The probability is then modified proportionnaly to the number of repetitions ##
        sampled_ensemble[Nid]['ProbScenBS'] = np.ones(TotBS)*1./Tot
        sampled_ensemble[Nid]['ProbScenPS'] = np.ones(TotPS)*1./Tot
        
        prob_angles_sum = np.sum(sampled_ensemble[Nid]['prob_scenarios_ang'])
        prob_angles_tot = sampled_ensemble[Nid]['prob_scenarios_ang']/prob_angles_sum

        ######### Duplication of scenarios beginning ##########
        ## The numbers of dupplicated scenarios are saved, then mutliplied by their respective probability 
        ## and the duplicates erased from the ensemble

        sample_unique_bs, test, counts_bs = np.unique(sampled_ensemble[Nid]['iscenbs'],return_index=True,return_counts=True) 
        unique_par = sampled_ensemble[Nid]['par_scenarios_bs'][test,:]
        unique_fact = sampled_ensemble[Nid]['prob_scenarios_bs_fact'][test,:]
        unique_prob = sampled_ensemble[Nid]['ProbScenBS'][test]
        unique_ang = sampled_ensemble[Nid]['prob_scenarios_ang'][test]
        unique_name = sampled_ensemble[Nid]['iscenbs'][test]
        ProbScenBS = np.ones(len(unique_prob))*1./Tot
        for itmp in range(len(unique_prob)):
           iscenbs=unique_name[itmp]
           indexbs=np.where(sample_unique_bs == iscenbs)
           if Kagan_weights>0:
              ProbScenBS[itmp]=unique_prob[itmp]*counts_bs[indexbs]/unique_ang[itmp]
           else:
              ProbScenBS[itmp]=unique_prob[itmp]*counts_bs[indexbs]#/unique_ang[itmp]

        sampled_ensemble[Nid]['par_scenarios_bs'] = unique_par
        sampled_ensemble[Nid]['prob_scenarios_bs_fact'] = unique_fact
        sampled_ensemble[Nid]['ProbScenBS'] = ProbScenBS/np.sum(ProbScenBS)
        sampled_ensemble[Nid]['relevant_scenarios_bs'] = np.unique(sampled_ensemble[Nid]['par_scenarios_bs'][:,0])
        ProbScenBS = sampled_ensemble[Nid]['ProbScenBS'] 

        sample_unique_ps, test, counts_ps = np.unique(sampled_ensemble[Nid]['iscenps'],return_index=True,return_counts=True)
        unique_par = sampled_ensemble[Nid]['par_scenarios_ps'][test,:]
        unique_fact = sampled_ensemble[Nid]['prob_scenarios_ps_fact'][test,:]
        unique_prob = sampled_ensemble[Nid]['ProbScenPS'][test]
        unique_name = sampled_ensemble[Nid]['iscenps'][test]
        ProbScenPS = np.ones(len(unique_prob))*1./Tot
        for itmp in range(len(unique_prob)):
           iscenps=unique_name[itmp]
           indexps=np.where(sample_unique_ps == iscenps)
           ProbScenPS[itmp]=unique_prob[itmp]*counts_ps[indexps]
        sampled_ensemble[Nid]['par_scenarios_ps'] = unique_par
        sampled_ensemble[Nid]['prob_scenarios_ps_fact'] = unique_fact
        sampled_ensemble[Nid]['ProbScenPS'] = ProbScenPS/np.sum(ProbScenPS)
        sampled_ensemble[Nid]['relevant_scenarios_ps'] = np.unique(sampled_ensemble[Nid]['par_scenarios_ps'][:,0])

        ### Re-normalization of all the probabilities ###
        TotProbBS = np.sum(ProbScenBS)
        TotProbPS = np.sum(ProbScenPS)
        TOT = TotProbPS+TotProbBS
        print('Probability',TotProbBS,TotProbPS,TOT)
        print(' --> Relevant Scenarios BS : ', sampled_ensemble[Nid]['relevant_scenarios_bs'])
        print(' --> Relevant Scenarios PS : ', sampled_ensemble[Nid]['relevant_scenarios_ps'])

        try:
            max_idxBS = np.argmax(ProbScenBS)
        except:
            max_idxBS = -1
        try:
            max_ValBS = ProbScenBS[max_idxBS]
        except:
            max_ValBS = 0
        try:
            max_idxPS = np.argmax(ProbScenPS)
        except:
            max_idxPS = -1
        try:
            max_ValPS = ProbScenPS[max_idxPS]
        except:
            max_ValPS = 0

            max_ValPS = 0

            max_ValPS = 0

        sampled_ensemble[Nid]['best_scenarios'] = {'max_idxBS':max_idxBS, 'max_idxPS':max_idxPS, 'max_ValBS':max_ValBS, 'max_ValPS':max_ValPS}

        print('     --> Best Bs scenario Idx and Value: %6d    %.5e' % (max_idxBS, max_ValBS))
        print('     --> Best Ps scenario Idx and Value: %6d    %.5e' % (max_idxPS, max_ValPS))
        
        Nid=Nid+1

    return sampled_ensemble

def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]

def set_if_compute_scenarios(**kwargs):

    short_term     = kwargs.get('short_term', None)
    Config         = kwargs.get('cfg', None)

    neg_prob       = float(Config.get('Settings','negligible_probability'))

    out = dict()

    # Some inizialization
    out['nr_ps_scenarios'] = 0
    out['nr_bs_scenarios'] = 0

    BScomputedYN   = False
    PScomputedYN   = False

    tmpbs      = (short_term['magnitude_probability'] * short_term['RatioBSonTot']).sum()
    tmpps      = (short_term['magnitude_probability'] * short_term['RatioPSonTot']).sum()

    if(tmpbs > neg_prob):
        BScomputedYN = True
    if(tmpps > neg_prob):
        PScomputedYN = True

    out['BScomputedYN'] = BScomputedYN
    out['PScomputedYN'] = PScomputedYN

    print(' <-- Negligible Probability: %.4f' % neg_prob)
    print(' --> Sum of probability BS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpbs, BScomputedYN))
    print(' --> Sum of probability PS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpps, PScomputedYN))

    return out

def bs_probability_scenarios(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    samp_ens        = kwargs.get('samp_ens', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)
    NBS	            = kwargs.get('NBS', None)
    Nid             = kwargs.get('Nid', None)
    int_ens         = kwargs.get('intervals_ensemble', None)
    region_info     = dict()

    MC_type = Config.get('Sampling','MC_type')

    if(samp_ens[Nid]['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):
        samp_ens[Nid]['nr_bs_scenarios'] = 0
        return samp_ens
    regions_nr = []

    ### Generation of an array (size of the new ensemble) of random probability 
    if MC_type=='MC':
       random_value = np.random.random(NBS)
    if MC_type=='LH':
       sampler = stats.qmc.LatinHypercube(d=1)
       random_value = sampler.random(n=NBS)
    
    iscenbs=0
    for i in random_value:
        ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
        idx,proba = find_nearest(int_ens,i)
        ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
        ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
        samp_ens[Nid]['iscenbs'][iscenbs]=idx
        samp_ens[Nid]['prob_scenarios_bs'][iscenbs]=prob_scenes['ProbScenBS'][idx]
        for j in range(5):
            samp_ens[Nid]['prob_scenarios_bs_fact'][iscenbs,j]=prob_scenes['prob_scenarios_bs_fact'][idx,j]
        for j in range(11):
            samp_ens[Nid]['par_scenarios_bs'][iscenbs,j]=prob_scenes['par_scenarios_bs'][idx,j] 
 
        # Inside the original code the strike/dip/rake
        # do not depend of the magnitude and position
        latlon_1=prob_scenes['par_scenarios_bs'][idx,2]
        latlon_0=prob_scenes['par_scenarios_bs'][idx,3]
        bs2_pos = len(pre_selection['BS2_Position_Selection_inn'])
        d_latlon=np.zeros((bs2_pos,2))
        d_diff=np.zeros((bs2_pos))
        for val in range(bs2_pos):
                tmp_idx = pre_selection['BS2_Position_Selection_inn'][val]
                d_latlon[val,:] = Discretizations['BS-2_Position']['Val'][tmp_idx].split()
                d_diff[val] = haversine(latlon_1, latlon_0, d_latlon[val,0], d_latlon[val,1])
        ipos_idx = int(np.argmin(d_diff))
        ipos = pre_selection['BS2_Position_Selection_inn'][ipos_idx]
        ireg = Discretizations['BS-2_Position']['Region'][ipos] #prob_scenes['par_scenarios_bs'][idx,0] #Discretizations['BS-2_Position']['Region'][ipos]
        # Faccio il load della regione se non già fatto
        if(ireg not in regions_nr):
            #print("...............................", region_files)
            region_info = load_region_infos(ireg         = ireg,
                                            region_info  = region_info,
                                            region_files = region_files)
            regions_nr.append(ireg)
        RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
        # Non credo che qui ci saranno errori, nel senso che i npy sono creati a partire dai mat conteneti roba
        if(RegMeanProb_BS4.size == 0):
             print(' --> WARNING: region info %d is empty!!!' % (ireg) )
        ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
        tmpProbAngles = RegMeanProb_BS4[ipos_reg[0]]
        id_sel = 0
        val_angles = prob_scenes['par_scenarios_bs'][idx,5]+100*prob_scenes['par_scenarios_bs'][idx,6]+10000*prob_scenes['par_scenarios_bs'][idx,7]
        for angles_id in range(len(Discretizations['BS-4_FocalMechanism']['Val'])):
            str_val,dip_val,rak_val = Discretizations['BS-4_FocalMechanism']['Val'][angles_id].split()
            val_check = float(str_val)+100*float(dip_val)+10000*float(rak_val)
            if abs(val_angles-val_check)<0.0001:
               #print("One matching value for:",iscenbs,angles_id,tmpProbAngles[angles_id])
               #print(str_val,dip_val,rak_val)
               id_sel=angles_id
        ProbAngles = tmpProbAngles[id_sel]
        samp_ens[Nid]['prob_scenarios_ang'][iscenbs]=ProbAngles

        iscenbs=iscenbs+1

    samp_ens[Nid]['nr_bs_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_bs_fact'])[0]
    return samp_ens


def ps_probability_scenarios(**kwargs):

    Config             = kwargs.get('cfg', None)
    short_term         = kwargs.get('short_term', None)
    prob_scenes        = kwargs.get('prob_scenes', None)
    samp_ens           = kwargs.get('samp_ens', None)
    pre_selection      = kwargs.get('pre_selection', None)
    Model_Weights      = kwargs.get('Model_Weights', None)
    regions            = kwargs.get('regions', None)
    PSBarInfo          = kwargs.get('PSBarInfo', None)
    region_ps          = kwargs.get('region_ps', None)
    Scenarios_PS       = kwargs.get('Scenarios_PS', None)
    ps1_magnitude      = kwargs.get('ps1_magnitude', None)
    lambda_bsps        = kwargs.get('lambda_bsps', None)
    NPS                = kwargs.get('NPS', None)
    Nid                = kwargs.get('Nid', None)
    int_ens            = kwargs.get('intervals_ensemble', None)

    samp_ens[Nid]['PScomputedYN'] == False
    
    if(samp_ens[Nid]['PScomputedYN'] == False or short_term['PS_computed_YN'] == False):

        #print("--------uuuuuuuuuuuu------------>>", samp_ens[Nid]['PScomputedYN'], short_term['PS_computed_YN'])

        #fbfix 2021-11-26
        samp_ens[Nid]['PScomputedYN']    = False
        short_term['PS_computed_YN']   = False
        #
        samp_ens[Nid]['nr_ps_scenarios'] = 0

        return samp_ens

    ### Generation of an array (size of the new ensemble) of random probability 
    random_value = np.random.random(NPS)
    iscenps=0
    for i in random_value:
        ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
        idx,proba = find_nearest(int_ens,i)
        ### samp_ens corresponds to the new ensemble where the identification nb of each scenario in 
        ### the initial ensemble is saved in iscenbs, and the parameters and the probability as well
        samp_ens[Nid]['iscenps'][iscenps]=idx
        samp_ens[Nid]['prob_scenarios_ps'][iscenps]=prob_scenes['ProbScenPS'][idx]
        for j in range(5):
            samp_ens[Nid]['prob_scenarios_ps_fact'][iscenps,j]=prob_scenes['prob_scenarios_ps_fact'][idx,j]
        for j in range(7):
            samp_ens[Nid]['par_scenarios_ps'][iscenps,j]=prob_scenes['par_scenarios_ps'][idx,j]
        iscenps=iscenps+1

    samp_ens[Nid]['nr_ps_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_ps_fact'])[0]

    return samp_ens


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()

    region_info[ireg] = info

    return region_info

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

