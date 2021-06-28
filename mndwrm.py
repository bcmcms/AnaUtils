#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
### Compiled with Keras-2.3.1 Tensorflow-1.14.0                      ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

# from ROOT import TH1F, TFile, gROOT, TCanvas

import sys, math
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist, PhysObj, Event, inc, fstrip, InputConfig, dphi
from analib import dframe as DataFrame
import pickle
import copy as cp
#from uproot_methods import TLorentzVector, TLorentzVectorArray
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.keras import backend as BE
from keras import backend as K
import json
import threading
from uproot_methods import TLorentzVectorArray as TLVA

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)


import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()

pd.options.mode.chained_assignment = None

##Controls how many epochs the network will train for
epochs = 50
##The weights LHE segment split data should be merged by
#lheweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401] # old bg
#lheweights = [1.0,0.33,0.034,0.034,0.024,0.0024,0.00044] #bEnriched bg
#lheweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401,1.0,0.33,0.034,0.034,0.024,0.0024,0.00044]
#nlhe = len(lheweights)
##Controls whether a network is trained up or loaded from disc
#LOADMODEL = True
##Switches tutoring mode on or off
#TUTOR = False
##Switches whether training statistics are reported or suppressed (for easier to read debugging)
#VERBOSE=False
##Switches whether weights are loaded and applied to the post-training statistics,
##and what data file they expect to be associated with
#POSTWEIGHT = True
JSONNAME = 'C2018.json'
bgdbg, sigdbg = '',''

#evtlist = [35899001,24910172,106249475,126514437,43203653,27186346,17599588,64962950,61283040,54831588]

def binary_focal_loss(alpha=.25, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def tutor(X_train,X_test,Y_train,Y_test):
    records = {}
    rsums = {}
    
    def tutorthread(l1,l2,l3,alpha,gamma,lr,records,rsums):
        rname = str(l1)+' '+str(l2)+' '+str(l3)+': alpha '+str(alpha)+' gamma '+str(gamma)
        aoc = []
        for i in range(10):
            #tf.random.set_random_seed(2)
            #tf.compat.v1.random.set_random_seed(2)
            #tf.compat.v1.set_random_seed(2)
            np.random.seed(2)
            model = keras.Sequential([
                #keras.Input(shape=(4,),dtype='float32'),
                #keras.layers.Flatten(input_shape=(8,)),
                keras.layers.Dense(l1, activation=tf.nn.relu,input_shape=(len(X_test.shape[1]),)),
                keras.layers.Dense(l2, activation=tf.nn.relu),
                keras.layers.Dense(l3, activation=tf.nn.relu),
                #keras.layers.Dropout(0.1),
                keras.layers.Dense(1, activation=tf.nn.sigmoid),
                ])
            optimizer  = keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer,     
                          #loss='binary_crossentropy',
                          #loss=[focal_loss],
                          #loss=[custom],
                          loss=[binary_focal_loss(alpha, gamma)],
                          metrics=['accuracy'])#,tf.keras.metrics.AUC()])

            model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True)

            rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
            aoc.append(auc(rocx,rocy))
        aocsum = np.sum(aoc)/10               
        records[rname] = aocsum
        rsums[rname] = aoc
        print(rname,' = ',aocsum)
        
    lr=.01
    threadlist = []
    for l1 in [8,12,24,36]:
        for l2 in [8,12,24,36]:
            for l3 in [8,12,24,36]:            
                for alpha in [0.5,0.6,0.7,0.8,0.85,0.9]:
                    for gamma in [0.6,0.7,0.8,0.85,0.9,1.0,1.2]:
                        threadlist.append(threading.Thread(target=tutorthread,args=(l1,l2,l3,alpha,gamma,lr,records,rsums)))
                        threadlist[-1].start()
                        
    for th in threadlist:
        th.join()
                        
                    
    winner = max(records,key=records.get)
    print('Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    winner = max(records,key=records.get)
    print('2nd Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    winner = max(records,key=records.get)
    print('3rd Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    sys.exit()

def pucalc(evv,hlt):
    pvals = uproot.open('PU_ratio_2021_05_26.root').get('PU_ratio').values
    fvals = uproot.open('PU_ratio_2021_05_26.root').get('PU_ratio_HLT_AK8PFJet330').values
    for v in range(99):
        evv.extweight[(evv.PU == v) & hlt.AK8PFJet500 & (evv.trg != "CX")] *= pvals[v]
        evv.extweight[(evv.PU == v) & ~(hlt.AK8PFJet500 & (evv.trg != "CX"))] *= fvals[v]
    evv.extweight[(evv.PU >= 99) & hlt.AK8PFJet500 & (evv.trg != "CX")] *= pvals[99]   
    evv.extweight[(evv.PU >= 99) & ~(hlt.AK8PFJet500 & (evv.trg != "CX"))] *= fvals[99]

def fastdR(solobj,multobj):
    seta, sphi = DataFrame(), DataFrame()
    
    for c in multobj.eta.columns:
        seta[c] = solobj.eta[1]
        sphi[c] = solobj.phi[1]
        
    dr2 = (seta - multobj.eta)**2 + dphi(sphi, multobj.phi)**2
    return dr2**.5

def computedR(jet,thing,nms=['jet','thing']):
    print("starting to compute dR")
    
    nj = jet.eta.columns
    nt = thing.eta.columns
    ## Create our dR dataframe by populating its first column and naming it accordingly
    if 1 in nj and 1 in nt:
        jtdr2 = DataFrame(np.power(jet.eta[1] - thing.eta[1],2) + np.power(dphi(jet.phi[1],thing.phi[1]),2)).rename(columns={1:f"{nms[0]} 1 {nms[1]} 1"})
    else: jtdr2 = DataFrame()
    jtstr = []
    ## Loop over jet x thing combinations
    for j in nj:
        for t in nt:
            jtstr.append(f"{nms[0]} {j} {nms[1]} {t}")
            if (j+t==2):
                continue
            jtdr2[jtstr[-1]] = pd.Series(np.power(jet.eta[j]-thing.eta[t],2) + np.power(dphi(jet.phi[j],thing.phi[t]),2))
    return np.sqrt(jtdr2)

def loadjets(jets, ev, events,bgweights=False):
    jets.eta= DataFrame(events.array('FatJet_eta')).rename(columns=inc)
    jets.phi= DataFrame(events.array('FatJet_phi')).rename(columns=inc)
    jets.pt = DataFrame(events.array('FatJet_pt')).rename(columns=inc)
    jets.mass=DataFrame(events.array('FatJet_mass')).rename(columns=inc)
    jets.CSVV2 = DataFrame(events.array('FatJet_btagCSVV2')).rename(columns=inc)
    jets.DeepB = DataFrame(events.array('FatJet_btagDeepB')).rename(columns=inc)
    jets.DDBvL = DataFrame(events.array('FatJet_btagDDBvL')).rename(columns=inc)
    jets.msoft = DataFrame(events.array('FatJet_msoftdrop')).rename(columns=inc)
    jets.H4qvs = DataFrame(events.array('FatJet_deepTagMD_H4qvsQCD')).rename(columns=inc)
    jets.n2b1  = DataFrame(events.array('FatJet_n2b1')).rename(columns=inc)
    jets.n2b1[jets.n2b1 < -5] = -2

    ev.event = DataFrame(events.array('event')).rename(columns=inc)
    ev.npvs  = DataFrame(events.array('PV_npvs')).rename(columns=inc)
    ev.npvsG = DataFrame(events.array('PV_npvsGood')).rename(columns=inc)
    ev.trg   = DataFrame(events.array('event')).rename(columns=inc)
    ev.trg[1]= 'X'
    if 'Pileup_nTrueInt' in events:
        ev.PU    = DataFrame(events.array('Pileup_nTrueInt')).rename(columns=inc)
    
    idxa1 = events.array('FatJet_subJetIdx1')
    idxa2 = events.array('FatJet_subJetIdx2')
    submass = events.array('SubJet_mass')
    subtau  = events.array('SubJet_tau1')
    submass = submass.pad(submass.counts.max()+1).fillna(0)
    subtau  = subtau.pad(subtau.counts.max()+1).fillna(0)
    jets.submass1 = DataFrame(submass[idxa1]).rename(columns=inc)
    jets.submass2 = DataFrame(submass[idxa2]).rename(columns=inc)
    jets.subtau1  = DataFrame(subtau[ idxa1]).rename(columns=inc)
    jets.subtau2  = DataFrame(subtau[ idxa2]).rename(columns=inc)
    del idxa1, idxa2, submass, subtau
    
    ev.extweight = ev.event / ev.event
    if bgweights:
        ev.HT = DataFrame(events.array('LHE_HT')).rename(columns=inc)
        
        pdgid = DataFrame(events.array('GenPart_pdgId')).rename(columns=inc)
        pdgpt = DataFrame(events.array('GenPart_pt')).rename(columns=inc)
        pdgframe = np.logical_and(pdgid == 5, pdgpt > 15)
        ev.nGprt = DataFrame(pdgframe[pdgframe].sum(axis=1)).rename(columns=inc)
        
    else:
        ev.HT = ev.event * 6000 / ev.event

    return jets, ev

def trigtensorcalc(jets,sj,evv,l1,hlt,isdata=False,dropjets=True):
    
    for attempt in range(2):
        try:        
            sjets = sj.deepcopy()
            ttensor = pickle.load(open('TrigTensor.p','rb'))
            maxjets = jets.deepcopy()
            if len(maxjets.columns) > 1:
                maxjets.cut(jets['pt'].rank(axis=1,method='first',ascending=False) == 1)
                for elem in ['pt','eta','phi']:
                    maxjets[elem] = DataFrame(maxjets[elem].max(axis=1)).rename(columns=inc)
            sjets.cut(abs(sjets.eta) < 2.4,drop=dropjets)
            # sjets.cut(sjets.pt > 30)
            sjets.cut(sjets.puId >= 1,drop=dropjets)
            sjets.cut(sjets.pt > 140,drop=dropjets)
            sjets.cut(sjets.btagDeepB > 0.4184,drop=dropjets)
            if sjets.pt.size == 0:
                raise ValueError("AK4 jets were cut to nonexistence by trigtensorcalc")
        except ValueError:
            dropjets=False
        else:
            break
            
    sjjdr = fastdR(maxjets,sjets)#computedR(maxjets,sjets,['Fatjet','slimjet'])
    # jlist = [sjjdr.filter(like=f"Fatjet {1}")]
    # jlist[0] = jlist[0].rename(columns=lambda x:int(x[-2:None]))
    #jlist[0][jlist[0] == 0] = jlist[0]+0.001
    sjframe = sjjdr[sjjdr < 0.8]#jlist[0][jlist[0] < 0.8]#.fillna(0)
    # for j in maxjets.pt.columns[1:]:
    #     jlist[j][jlist[j] == 0] = jlist[j]+0.001
    #     sjframe = sjframe + jlist[j][jlist[j] < 0.8].fillna(0)
    njets = sjframe.rank(axis=1,method='first',ascending=False).max(axis=1).fillna(0)
    # sj2max = sjframe[sjframe.rank(axis=1,method='first',ascending=False) == 2]
    sj2maxb = sjets.btagDeepB[sjframe > 0]
    sj2maxb = sj2maxb[sj2maxb.rank(axis=1,method='first',ascending=False) == 2].max(axis=1).reindex_like(evv.trg)
    
    evv.trg[(maxjets.pt[1] < 400) & (maxjets.pt[1] >= 250) & (njets >= 2) &
        (l1.DoubleJet112er2p3_dEta_Max1p6[1] | l1.DoubleJet150er2p5[1]) &
        hlt.DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71[1]] = "CX"
    
    evv.trg[(maxjets.pt[1] >= 400) & (njets < 2) &
        (hlt.AK8PFJet500[1] | hlt.AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4[1]) &
        l1.SingleJet180[1]] = "AB"
    
    evv.trg[(maxjets.pt[1] >= 400) & (njets >= 2) &
        ((l1.SingleJet180[1] & (hlt.AK8PFJet500[1] | hlt.AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4[1])) |
        (hlt.DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71[1] &
            (l1.DoubleJet112er2p3_dEta_Max1p6[1] | l1.DoubleJet150er2p5[1])))] = "ABC"
    
    if not isdata:
        idxs = ttensor['meta']['AB']
        for i in range(len(idxs) - 1):
            evv.extweight[(evv.trg[1] == "AB") & (maxjets.pt[1] >= idxs[i]) & (maxjets.pt[1] < idxs[i+1])] *= ttensor["AB"][i]
        idxs = ttensor['meta']['ABC']
        for i in range(len(idxs) - 1):
            evv.extweight[(evv.trg[1] == "ABC") & (maxjets.pt[1] >= idxs[i]) & (maxjets.pt[1] < idxs[i+1])] *= ttensor["ABC"][i]
        idxs = ttensor['meta']['CX']
        for i in range(len(idxs) - 1):
            evv.extweight[(evv.trg[1] == "CX") & (sj2maxb >= idxs[i]) & (sj2maxb < idxs[i+1])] *= ttensor["CX"][i]     
    evv.cut(evv.trg != "X")
    # evv.extweight[evv.trg[1] == "X"] *= 0


#%%

def ana(sigfile,bgfile,LOADMODEL=True,TUTOR=False,passplots=False):
    #%%
    ic = InputConfig(sigfile,bgfile)
    if ic.sigdata:
        dataflag = 1
        if ic.bgdata:
            raise ValueError("Both signal and background are marked as data")
    elif ic.bgdata:
        dataflag = -1
    else: dataflag = 0
    sigfiles = ic.sigfiles
    bgfiles = ic.bgfiles
    if ic.bglhe:
        isLHE=True
        lheweights = ic.bgweight
        nlhe = len(lheweights)
        if not ic.siglhe:
            lheweights = np.divide(ic.bgweight, ic.size)
    else: isLHE=False
    
    netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','n2b1','submass1','submass2','subtau1','subtau2','nsv']
    l1vars = ['SingleJet180','Mu7_EG23er2p5','Mu7_LooseIsoEG20er2p5','Mu20_EG10er2p5','SingleMu22',
              'SingleMu25','DoubleJet112er2p3_dEta_Max1p6','DoubleJet150er2p5']
    hltvars = ['AK8PFJet500','Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ','Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
               'Mu27_Ele37_CaloIdL_MW','Mu37_Ele27_CaloIdL_MW',
               'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
               'DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71']
    slimvars = ['pt','eta','phi','btagDeepB','puId']
    

    ###################
    # Plots and Setup #
    ###################
    plargs = {'Data':  {'color':'k','htype':'step'},
              'Background':   {'color':'r','htype':'step'},
              'Signal':{'color':'b','htype':'step'}
              }
    if dataflag == True:
        Skey = 'Data'
    else: Skey = 'Signal'
    if dataflag == -1:
        Bkey = 'Data'
    else: Bkey = 'Background'
    if ic.signame:
        Sname = ic.signame
    else: Sname = Skey
    if ic.bgname:
        Bname = ic.bgname
    else: Bname = Bkey
    ## The dataflag controls which file list if any has been replaced by data
    if dataflag:
        LOADMODEL = True
        if dataflag == True:
            plargs['Data'].update({'htype':'err'})
            plargs['Background'].update( {'htype':'bar'})
    if TUTOR:
        LOADMODEL = False
    if not LOADMODEL:
        passplots = False

    l1 = 8
    l2 = 8
    l3 = 8
    alpha = 0.5
    gamma = 2.2
    model = keras.Sequential([
            keras.layers.Dense(l1, activation=tf.nn.relu,input_shape=(len(netvars),)),
            keras.layers.Dense(l2, activation=tf.nn.relu),
            keras.layers.Dense(l3, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ])
    optimizer  = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,    
                  loss=[binary_focal_loss(alpha, gamma)],
                  metrics=['accuracy'])
    scaler = MinMaxScaler()
    
    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
    plots = {
        "Distribution": Hist(50,(0,1),'Confidence','Fraction of Events','netplots/Distribution'),
        "DistributionL": Hist(50,(0,1),'Confidence','Fraction of Events','netplots/LogDistribution'),
        "DistStr":  Hist(50,(0,1)),
        "DistSte":  Hist(50,(0,1)),
        "DistBtr":  Hist(50,(0,1)),
        "DistBte":  Hist(50,(0,1)),
        "SensS"  :  Hist(10,(0,1)),
        "SensB"  :  Hist(10,(0,1), 'Confidence','Fraction of Events','netplots/Sensitivity'),
        "LossvEpoch":   Hist(epochs,(0.5,epochs+.5),'Epoch Number','Loss','otherplots/LossvEpoch'),
        "AccvEpoch":Hist(epochs,(0.5,epochs+.5),'Epoch Number','Accuracy','otherplots/AccvEpoch'),
    }
    
    pplots = {
        "pt":       Hist(80 ,(150,550)  ,'pT for highest pT jet','Fractional Distribution','netplots/ppt'),
        "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT jet','Fractional Distribution','netplots/peta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/pphi'),
        "mass":     Hist(32 ,(85,205)   ,'mass for highest pT jet','Fractional Distribution','netplots/pmass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet in all (red), passing signal (blue), and signal (black) events','Fractional Distribution','netplots/pCSVV2'),
        "DeepB":    Hist(14 ,(0.35,1.05),'DeepB for highest pT jet','Fractional Distribution','netplots/pDeepB'),
        "msoft":    Hist(32 ,(85,205)   ,'msoft for highest pT jet in all (red), passing (blue), and failing  (black) events','Fractional Distribution','netplots/pmsoft'),
        "DDBvL":    Hist(26 ,(0.76,1.02),'DDBvL for highest pT jet','Fractional Distribution','netplots/pDDBvL'),
        "H4qvs":    Hist(24 ,(-10,2)    ,'H4qvs for highest pT jet','Fractional Distribution','netplots/pH4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/pnpvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/pnpvsG'),
        "mpt":      Hist(25 ,(5.5,30.5) ,'pT for highest pT muon','Fractional Distribution','netplots/pmpt'),
        "meta":     Hist(15 ,(0,3)      ,'|eta| for highest pT muon','Fractional Distribution','netplots/pmeta'),
        "mip":      Hist(29 ,(1.5,30.5) ,'dxy/dxyError for highest pT muon','Fractional Distribution','netplots/pmip'),
        "HT":       Hist(500 ,(0,5000)  ,'pT for highest pT jet','Fractional Distribution','netplots/pHT'),
        "n2b1":     Hist(10 ,(0,1)      ,'n2b1 for highest pT jet','Fractional Distribution','netplots/pn2b1'),
        "submass1": Hist(105,(-5,105)   ,'submass for 1st subjet of highest pT jet','Fractional Distribution','netplots/psubmass1'),
        "submass2": Hist(105,(-5,105)   ,'submass for 2nd subjet of highest pT jet','Fractional Distribution','netplots/psubmass2'),
        "subtau1":  Hist(10 ,(0,1)      ,'subtau1 for 1st subjet of highest pT jet','Fractional Distribution','netplots/psubtau1'),
        "subtau2":  Hist(10 ,(0,1)      ,'subtau1 for 2nd subjet of highest pT jet','Fractional Distribution','netplots/psubtau2'),
        'nsv':      Hist(12 ,(-1,11)    ,'# of secondary vertices with dR<0.8 to highest pT jet','Fractional Distribution','netplots/pnsv'),
        }
    
    prefix = ['SG','SPS','SFL','BG','BPS','BFL']
    tdict = {}
    for plot in pplots:
        size = pplots[plot].size
        bounds = pplots[plot].bounds
        for fix in prefix:
            tdict.update({fix+plot:Hist(size,bounds)})
        tdict.update({'B'+plot:cp.deepcopy(pplots[plot])})
        tdict['B'+plot].fname = tdict['B'+plot].fname+'B'
    pplots.update(tdict)
    del tdict
    
    vplots = {
        "pt":       Hist(30 ,(250,850)  ,'pT for highest pT jet','Fractional Distribution','netplots/pt'),
        "eta":      Hist(30 ,(-3,3)      ,'|eta| for highest pT jet','Fractional Distribution','netplots/eta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/phi'),
        "mass":     Hist(32 ,(85,205)   ,'mass for highest pT jet','Fractional Distribution','netplots/mass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet','Fractional Distribution','netplots/CSVV2'),
        "DeepB":    Hist(14 ,(0.35,1.05),'DeepB for highest pT jet','Fractional Distribution','netplots/DeepB'),
        "msoft":    Hist(32 ,(85,205)   ,'msoft for highest pT jet','Fractional Distribution','netplots/msoft'),
        "DDBvL":    Hist(26 ,(0.76,1.02),'DDBvL for highest pT jet','Fractional Distribution','netplots/DDBvL'),
        "H4qvs":    Hist(20 ,(0,1)      ,'H4qvs for highest pT jet','Fractional Distribution','netplots/H4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/npvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/npvsG'),
        "mpt":      Hist(25 ,(5.5,30.5) ,'pT for highest pT muon','Fractional Distribution','netplots/mpt'),
        "meta":     Hist(15 ,(0,3)      ,'|eta| for highest pT muon','Fractional Distribution','netplots/meta'),
        "mip":      Hist(29 ,(1.5,30.5) ,'dxy/dxyError for highest pT muon','Fractional Distribution','netplots/mip'),
        "HT":       Hist(500,(0,5000)   ,'pT for highest pT jet','Fractional Distribution','netplots/HT'),
        "n2b1":     Hist(20 ,(0,1)      ,'n2b1 for highest pT jet','Fractional Distribution','netplots/n2b1'),
        "submass1": Hist(22 ,(-5,105)   ,'submass for 1st subjet of highest pT jet','Fractional Distribution','netplots/submass1'),
        "submass2": Hist(22 ,(-5,105)   ,'submass for 2nd subjet of highest pT jet','Fractional Distribution','netplots/submass2'),
        "subtau1":  Hist(20 ,(0,.5)     ,'subtau1 for 1st subjet of highest pT jet','Fractional Distribution','netplots/subtau1'),
        "subtau2":  Hist(20 ,(0,.5)     ,'subtau1 for 2nd subjet of highest pT jet','Fractional Distribution','netplots/subtau2'),
        'nsv':      Hist(12 ,(-1,11)    ,'# of secondary vertices with dR<0.8 to highest pT jet','Fractional Distribution','netplots/nsv'),
        
    }
    # mplots = {
    #     'mmsum':    Hist(7  ,(0,70)     ,'Mass of summed muons','Fractional Distribution','netplots/mmsum'),
    #     'mptsum':   Hist(110,(0,550)    ,'pT of summed muons','Fractional Distribution','netplots/mptsum'),
    #     'metasum':  Hist(15 ,(0,3)      ,'|eta| of summed muons','Fractional Distribution','netplots/metasum'),
    #     'mqmpt':    Hist(60 ,(0,300)    ,'pt of - muon','Fractional Distribution','netplots/mqmpt'),
    #     'mqppt':    Hist(60 ,(0,300)    ,'pt of + muon','Fractional Distribution','netplots/mqppt'),
    #     'mqmeta':   Hist(15 ,(0,3)      ,'|eta| of - muon','Fractional Distribution','netplots/mqmeta'),
    #     'mqpeta':   Hist(15 ,(0,3)      ,'|eta| of + muon','Fractional Distribution','netplots/mqpeta'),
    #     'mqmip3d':  Hist(20 ,(0,1)      ,'ip3d of - muons','Fractional Distribution','netplots/mqmip3d'),
    #     'mqpip3d':  Hist(20 ,(0,1)      ,'ip3d of + muons','Fractional Distribution','netplots/mqpip3d'),
    #     'mqmsip3d': Hist(20 ,(0,1)      ,'sip3d of - muons','Fractional Distribution','netplots/mqmip3d'),
    #     'mqpsip3d': Hist(20 ,(0,1)      ,'sip3d of + muons','Fractional Distribution','netplots/mqpip3d'),
    #     'MuSumvJetpT':  Hist(20, (0,2),'summed muon pT / AK8 Jet pT','Fractional Distribution','netplots/MuSumvJetpT'),
    #     'MuQmvJetpT':   Hist(20, (0,2),'- muon pT / AK8 Jet pT','Fractional Distribution','netplots/MuQmvJetpT'),
    #     'MuQpvJetpT':   Hist(20, (0,2),'+ muon pT / AK8 Jet pT','Fractional Distribution','netplots/MuQpvJetpT'),
    #     }
    # vplots.update(mplots)
    tdict = {}
    prefix = ['BG','SG','RW']
    for plot in vplots:
        size = vplots[plot].size
        bounds = vplots[plot].bounds
        for fix in prefix:
            tdict.update({fix+plot:Hist(size,bounds)})
    vplots.update(tdict)
    del tdict
    
    if (dataflag):
        if dataflag == True:
            cutword = 'signal'
        else:
            cutword = 'background'
        def dataswap(plots):
            for key in plots:
                label = plots[key].xlabel.split(cutword)
                for i in range(1,len(label)):
                    label.insert(2*i-1,'data')
                plots[key].xlabel = ''.join(label)
            return plots
        pplots = dataswap(pplots)
        vplots = dataswap(vplots)
        plots  = dataswap(plots )
    
    if not LOADMODEL:
        for prop in ['mpt','meta','mip']:
            vplots.pop(prop)
            pplots.pop(prop)
                   
    if isLHE:
        lheplots = {}
        for i in range(nlhe):
            lheplots.update({'dist'+str(i):Hist(50,(0,1),'Normalized MC background classifcation','Fraction of Events','otherplots/LHEdist_'+str(i)),})
            lheplots['dist'+str(i)].title = 'Distrubution for LHE segment '+str(i)
#    for plot in plots:
#        plots[plot].title = files[0]
            

    if LOADMODEL:
        if dataflag == 1:
            pf = 'D'
            for key in vplots:
                vplots[key].ylabel = 'Events'
            for key in pplots:
                pplots[key].ylabel = 'Events'
        elif dataflag == -1:
            pf = 'C'
        else:
            pf = 'S'
        if passplots == True:
            pf = pf + 'P'
        for key in vplots:
            vplots[key].fname = pf+vplots[key].fname
        for key in pplots:
            pplots[key].fname = pf+pplots[key].fname
        for key in ['Distribution','DistributionL','SensB']:
            plots[key].fname = pf+plots[key].fname

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(ic.size):
        print('bg',len(bgfiles),'sig',len(sigfiles))
        
        #####################
        # Loading Variables #
        #####################
        if isLHE:
            print('Opening',sigfiles[fnum],'+ LHE Background')    
        else:
            print('Opening',sigfiles[fnum],'+',bgfiles[fnum])
            
        ## Open our file and grab the events tree
        if isLHE:
            bgevents = []
            for i in range(nlhe):
                print('Opening ',bgfiles[i])
                bgevents.append(uproot.open(bgfiles[i]).get('Events'))
        else:
            bgf = uproot.open(bgfiles[fnum])
            bgevents = bgf.get('Events')
        sigf = uproot.open(sigfiles[fnum])
        sigevents = sigf.get('Events')
        
        
             
        print(f"Dataflag:{dataflag}")

        sigjets, sigevv = loadjets(PhysObj('sigjets'),PhysObj("sigev"),sigevents)
        sigsv = PhysObj('sigsv',sigfiles[fnum],'eta','phi',varname='SV')
        sigl1 = PhysObj("sigl1",sigfiles[fnum], *l1vars, varname='L1')
        sighlt= PhysObj("sighlt",sigfiles[fnum], *hltvars,varname='HLT')
        ## slimjets are not included in automatic event vetos, they're for trigger region analysis
        sigsj = PhysObj("sigsj",sigfiles[fnum],*slimvars,varname='Jet')
        sigevv['extweight'] *= ic.sigweight[fnum]
        
            
        if isLHE:
            bgjets, bgsv, bgl1, bghlt, bgevv, bgsj = [],[],[],[],[],[]
            for i in range(nlhe):
                tjet, tevv = loadjets(PhysObj(f"bgjet{i}"),PhysObj(f"bgev{i}"),bgevents[i],True)
                bgjets.append(tjet)
                bgevv.append(tevv)
                bgsv.append(PhysObj(f"bgsv{i}"  ,bgfiles[i],'eta','phi',varname='SV'))
                bgl1.append(PhysObj(f"bgl1{i}"  ,bgfiles[i], *l1vars, varname='L1'))
                bghlt.append(PhysObj(f"bghlt{i}",bgfiles[i], *hltvars,varname='HLT'))
                bgsj.append(PhysObj(f"bgsj{i}"  ,bgfiles[i],*slimvars,varname='Jet'))
                bgevv[i].extweight *= lheweights[i]
                
                

                
        else:
            if dataflag == -1:
                bgjets, bgevv = loadjets(PhysObj('bgjets'),PhysObj("bgev"),bgevents)
            else:
                bgjets, bgevv = loadjets(PhysObj('bgjets'),PhysObj("bgev"),bgevents,True)
                bgevv['extweight'] *= ic.bgweight[fnum]
            bgjets = [bgjets]
            bgevv = [bgevv]
            bgsv = [PhysObj('bgsv',bgfiles[fnum],'eta','phi',varname='SV')]
            bgl1 = [PhysObj("bgl1",bgfiles[fnum], *l1vars, varname='L1')]
            bghlt= [PhysObj("bghlt",bgfiles[fnum], *hltvars,varname='HLT')]
            bgsv = [PhysObj('bgsv',bgfiles[fnum],'eta','phi',varname='SV')]
            bgsj = [PhysObj("bgsj",bgfiles[fnum],*slimvars,varname='Jet')]

        print(f"Processing {str(len(sigjets.eta))} {Skey} events")
        
        if (not dataflag) and (not LOADMODEL):
            pdgida  = sigevents.array('GenPart_pdgId')
            paridxa = sigevents.array('GenPart_genPartIdxMother')
            parida  = pdgida[paridxa] 
            
            bs = PhysObj('bs')
            
            ## Removes all particles that do not have A parents 
            ## from the GenPart arrays, then removes all particles 
            ## that are not bs after resizing the pdgid array to be a valid mask
            
            bs.oeta = DataFrame(sigevents.array('GenPart_eta')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.ophi = DataFrame(sigevents.array('GenPart_phi')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.opt  = DataFrame(sigevents.array('GenPart_pt' )[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            
            ## Test b order corresponds to As
            testbs = DataFrame(sigevents.array('GenPart_genPartIdxMother')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            ## The first term checks b4 has greater idx than b1, the last two check that the bs are paired
            if ((testbs[4]-testbs[1]).min() <= 0) or ((abs(testbs[2]-testbs[1]) + abs(testbs[4])-testbs[3]).min() != 0):
                print('b to A ordering violated - time to do it the hard way')
                sys.exit()
        
            As = PhysObj('As')    
            As.oeta = DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.ophi = DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.opt =  DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.omass =DataFrame(sigevents.array('GenPart_mass', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            
            higgs = PhysObj('higgs')    
            higgs.eta = DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.phi = DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.pt =  DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            
            slimjets = PhysObj('slimjets')
            slimjets.eta= DataFrame(sigevents.array('Jet_eta', executor=executor)).rename(columns=inc)
            slimjets.phi= DataFrame(sigevents.array('Jet_phi', executor=executor)).rename(columns=inc)
            slimjets.pt = DataFrame(sigevents.array('Jet_pt' , executor=executor)).rename(columns=inc)
            slimjets.mass=DataFrame(sigevents.array('Jet_mass', executor=executor)).rename(columns=inc)
            slimjets.DeepB = DataFrame(sigevents.array('Jet_btagDeepB', executor=executor)).rename(columns=inc)
            slimjets.DeepFB= DataFrame(sigevents.array('Jet_btagDeepFlavB', executor=executor)).rename(columns=inc)
            slimjets.puid = DataFrame(sigevents.array('Jet_puId', executor=executor)).rename(columns=inc)
        

            ## Figure out how many bs and jets there are
            nb = bs.oeta.shape[1]
            na = As.oeta.shape[1]
            if na != 2:
                print("More than two As per event, found "+str(na)+", halting")
                sys.exit()
            
            ## Create sorted versions of A values by pt
            for prop in ['eta','phi','pt','mass']:
                As[prop] = DataFrame()
                for i in range(1,3):
                    As[prop][i] = As['o'+prop][As.opt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
                ## Clean up original ordered dataframes; we don't really need them
                #del As['o'+prop]
            
            ## Reorder out b dataframes to match sorted A parents
            tframe = DataFrame()
            tframe[1] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[2] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[3] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            tframe[4] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            for prop in ['eta','phi','pt']:
                bs[prop] = DataFrame()
                bs[prop][1] = bs['o'+prop][tframe][1].dropna().append(bs['o'+prop][tframe][3].dropna()).sort_index()
                bs[prop][2] = bs['o'+prop][tframe][2].dropna().append(bs['o'+prop][tframe][4].dropna()).sort_index()
                bs[prop][3] = bs['o'+prop][~tframe][1].dropna().append(bs['o'+prop][~tframe][3].dropna()).sort_index()
                bs[prop][4] = bs['o'+prop][~tframe][2].dropna().append(bs['o'+prop][~tframe][4].dropna()).sort_index()
                ## Clean up original ordered dataframes; we don't really need them.
                #del bs['o'+prop]
            
#           ## Sort our b dataframes in descending order of pt
#           for prop in ['spt','seta','sphi']:
#               bs[prop] = DataFrame()
#           #bs.spt, bs.seta, bs.sphi = DataFrame(), DataFrame(), DataFrame()
#               for i in range(1,nb+1):
#                   bs[prop][i] = bs[prop[1:]][bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.seta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.sphi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            
#           plots['genAmass'].dfill(As.mass)

            ev = Event(bs,sigjets,As,higgs,sigsv)
        else:
            ev = Event(sigjets,sigsv,sigevv,sigl1,sighlt)
            
        
        bev = []
        for i in range(len(bgjets)):
            bev.append(Event(bgjets[i],bgsv[i],bgevv[i],bgl1[i],bghlt[i]))
            
        for jets in bgjets+[sigjets]:
            jets.cut(jets.pt > 170)
            jets.cut(abs(jets.eta)<2.4)
            jets.cut(jets.DDBvL > 0.8)
            jets.cut(jets.DeepB > 0.4184)
            jets.cut(jets.msoft > 90)
            jets.cut(jets.mass > 90)
            jets.cut(jets.mass < 200)
            jets.cut(jets.msoft < 200)
        for evv in bgevv + [sigevv]:
            evv.cut(evv.npvsG >= 1)
            
        for i in range(len(bgevv)):
            if ic.bgqcd[i+fnum*len(bgevv)] == -1:
                bgevv[i].cut(bgevv[i].nGprt == 0)
            elif ic.bgqcd[i+fnum*len(bgevv)]:        
                bgevv[i].cut(bgevv[i].nGprt >= 1)
                


        if (not dataflag) and (not LOADMODEL):
            bs.cut(bs.pt>5)
            bs.cut(abs(bs.eta)<2.4)
            ev.sync()
            slimjets.cut(slimjets.DeepB > 0.1241)
            slimjets.cut(slimjets.DeepFB > 0.277)
            slimjets.cut(slimjets.puid > 0)
            slimjets.trimto(sigjets.eta)
            
        #####################
        # Non-Training Cuts #
        #####################
            
        ## Apply golden JSON cuts to data events
        if dataflag == True:
            # events = sigevents
            # event = ev
        # elif dataflag == -1:
        #     events = bgevents
        #     event = bev
        # if dataflag != False:
            ## Apply data cuts
            with open(JSONNAME) as f:
                jdata = json.load(f)
            sigevv.run = DataFrame(sigevents.array('run')).rename(columns=inc)
            sigevv.lb = DataFrame(sigevents.array('luminosityBlock')).rename(columns=inc)
            ev.sync()

            ## Only events that are in keys are kept
            sigevv.cut(sigevv.run.isin(jdata.keys())==True)
            for elem in ['run','lb']:
                sigevv[elem]=sigevv[elem].astype(int)
            
            ## Test remaining events for inclusion
            tframe = DataFrame()
            tframe['lb'] = sigevv.lb[1]
            tframe['run'] = sigevv.run[1]
            def fun(r,lb):
                return any([lb in range(a[0],a[1]+1) for a in jdata[str(r)] ])
            truthframe = DataFrame([fun(r,lb) for r,lb in zip(sigevv.run[1].values,sigevv.lb[1].values)],index=sigevv.run.index,columns=[1])
            sigevv.cut(truthframe == True)
            ev.sync()

        ## Muon cuts for non-training comparisons
        if (LOADMODEL) and False:#(not passplots):
            muons = PhysObj('Muon',sigfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d')
            muons.ip = abs(muons.dxy / muons.dxyErr)
            muons.eta = abs(muons.eta)
            ev.register(muons)
            muons.cut(muons.softId > 0.9)
            muons.cut(abs(muons.eta) < 2.4)
            muons.cut(muons.pt > 7)
            muons.cut(muons.ip > 2)
            #muons.cut(muons.ip3d < 0.5)
            ev.sync()
            
            ## Apply these cuts to data events as well
            bmuons = []
            for i in range(len(bgjets)):
                idx = i#fnum*nlhe + i
                bmuons.append(PhysObj('Muon',bgfiles[idx],'softId','eta','pt','dxy','dxyErr','ip3d'))
                bmuons[i].eta = abs(bmuons[i].eta)
                bmuons[i].ip = abs(bmuons[i].dxy / bmuons[i].dxyErr)
                bev[i].register(bmuons[i])
                bmuons[i].cut(bmuons[i].softId > 0.9)
                bmuons[i].cut(abs(bmuons[i].eta) < 2.4)
                bmuons[i].cut(bmuons[i].pt > 7)
                bmuons[i].cut(bmuons[i].ip > 2)
                bmuons[i].cut(bmuons[i].ip3d < 0.5)
                bev[i].sync()
                


        
        ##########################
        # Training-Specific Cuts #
        ##########################
        if (not dataflag) and (not LOADMODEL):
            jbdr = computedR(sigjets,bs,['jet','b'])
            blist = []
            # sblist = []
            for b in range(nb):
                blist.append(jbdr.filter(like='b '+str(b+1)))
                blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
                blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))

            fjets = blist[0][blist[0]<0.8].fillna(0)/blist[0][blist[0]<0.8].fillna(0)
            for i in range(1,4):
                fjets = fjets + blist[i][blist[i]<0.8].fillna(0)/blist[i][blist[i]<0.8].fillna(0)
            fjets = fjets.max(axis=1)
            fjets = fjets[fjets==4].dropna()
            sigjets.trimto(fjets)
            ev.sync()
            del jbdr, blist, fjets
            
        ########################
        # Precompute Jet Crush #
        ########################
            
        for jets in bgjets + [sigjets]:
            jets.cut(jets.pt.rank(axis=1,method='first',ascending=False) == 1)
            for prop in jets:
                jets[prop] = DataFrame(jets[prop].max(axis=1)).rename(columns=inc)
        
        
        # ## Special region-selection muon cut logic
        # if (LOADMODEL) and True:#(not passplots):
        #     muons = PhysObj('Muon',sigfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d','charge','mass','phi','sip3d')
        #     muons.ip = abs(muons.dxy / muons.dxyErr)
        #     #muons.eta = abs(muons.eta)
        #     ev.register(muons)
        #     muons.cut(muons.softId > 0.9)
        #     muons.cut(abs(muons.eta) < 2.4)
        #     muons.cut(muons.pt > 5)
        #     #muons.cut(muons.ip > 2)
        #     muons.cut(abs(muons.ip3d) < 0.5)
        #     ev.sync()
            
        #     mframe = fastdR(sigjets, muons)#DataFrame()
        #     # mjdr = computedR(sigjets, muons,['jet','muon'])
        #     # mframe = mjdr.filter(like="jet 1")
        #     # mframe = mframe.rename(columns=lambda x:int(x[-2:None]))
        #     mframe = mframe[muons.pt != 0].dropna(how='all')
        #     muons.jetdr = mframe
        #     muons.cut(muons.jetdr < 0.8)
                
        #     ev.sync()
        #     pmuons, mmuons = muons.deepcopy(), muons.deepcopy()
        #     pmuons.cut(pmuons.charge == 1)
        #     mmuons.cut(mmuons.charge == -1)
        #     pmuons.cut(pmuons.mass.rank(axis=1,method='first',ascending=False) == 1)
        #     mmuons.cut(mmuons.mass.rank(axis=1,method='first',ascending=False) == 1)
        #     pmuons.trimto(mmuons.charge)
        #     mmuons.trimto(pmuons.charge)
        #     pTL = TLVA.from_ptetaphim(pmuons.pt.sum(axis=1),
        #             pmuons.eta.sum(axis=1),
        #             pmuons.phi.sum(axis=1),
        #             pmuons.mass.sum(axis=1))
        #     mTL = TLVA.from_ptetaphim(mmuons.pt.sum(axis=1),
        #             mmuons.eta.sum(axis=1),
        #             mmuons.phi.sum(axis=1),
        #             mmuons.mass.sum(axis=1))
        #     msum = DataFrame((pTL + mTL).mass).rename(columns=inc)
        #     ptsum = DataFrame((pTL + mTL).pt).rename(columns=inc)
        #     etasum = DataFrame((pTL + mTL).eta).rename(columns=inc)
        #     for frame in [msum, ptsum, etasum]:
        #         frame.index = mmuons.pt.index
        #     sigevv.mmsum = msum[msum > 12].dropna()
        #     sigevv.mptsum = ptsum.dropna()
        #     sigevv.metasum = etasum.dropna()
        #     sigevv.mqmpt = DataFrame(mmuons.pt.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqppt = DataFrame(pmuons.pt.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqmeta = DataFrame(mmuons.eta.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqpeta = DataFrame(pmuons.eta.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqmip3d = DataFrame(mmuons.ip3d.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqpip3d = DataFrame(pmuons.ip3d.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqmsip3d = DataFrame(mmuons.sip3d.sum(axis=1)).rename(columns=inc)
        #     sigevv.mqpsip3d = DataFrame(pmuons.sip3d.sum(axis=1)).rename(columns=inc)

        #     ev.sync()
        #     del pmuons, mmuons, msum, pTL, mTL
            
        #     ## Apply these cuts to data events as well
            
        #     bmuons = []
        #     for i in range(len(bgjets)):
        #         idx = i#fnum*nlhe + i
        #         bmuons.append(PhysObj('Muon',bgfiles[idx],'softId','eta','pt','dxy','dxyErr','ip3d','charge','mass','phi','sip3d'))
        #         #bmuons[i].eta = abs(bmuons[i].eta)
        #         bmuons[i].ip = abs(bmuons[i].dxy / bmuons[i].dxyErr)
        #         bev[i].register(bmuons[i])
        #         bmuons[i].cut(bmuons[i].softId > 0.9)
        #         bmuons[i].cut(abs(bmuons[i].eta) < 2.4)
        #         bmuons[i].cut(bmuons[i].pt > 5)
        #         #bmuons[i].cut(bmuons[i].ip > 2)
        #         bmuons[i].cut(abs(bmuons[i].ip3d) < 0.5)
        #         bev[i].sync()
                
        #         mframe = fastdR(bgjets[i], bmuons[i])#DataFrame()
        #         # mjdr = computedR(bgjets[i], bmuons[i],['jet','muon'])
        #         # mframe = mjdr.filter(like="jet 1")
        #         # mframe = mframe.rename(columns=lambda x:int(x[-2:None]))
        #         mframe = mframe[bmuons[i].pt != 0].dropna(how='all')
        #         bmuons[i].jetdr = mframe
        #         bmuons[i].cut(bmuons[i].jetdr < 0.8)
                
        #         pmuons, mmuons = bmuons[i].deepcopy(), bmuons[i].deepcopy()
        #         pmuons.cut(pmuons.charge == 1)
        #         mmuons.cut(mmuons.charge == -1)
        #         pmuons.cut(pmuons.mass.rank(axis=1,method='first',ascending=False) == 1)
        #         mmuons.cut(mmuons.mass.rank(axis=1,method='first',ascending=False) == 1)
        #         pmuons.trimto(mmuons.charge)
        #         mmuons.trimto(pmuons.charge)
        #         pTL = TLVA.from_ptetaphim(pmuons.pt.sum(axis=1),
        #                 pmuons.eta.sum(axis=1),
        #                 pmuons.phi.sum(axis=1),
        #                 pmuons.mass.sum(axis=1))
        #         mTL = TLVA.from_ptetaphim(mmuons.pt.sum(axis=1),
        #                 mmuons.eta.sum(axis=1),
        #                 mmuons.phi.sum(axis=1),
        #                 mmuons.mass.sum(axis=1))
        #         msum = DataFrame((pTL + mTL).mass).rename(columns=inc)
        #         ptsum = DataFrame((pTL + mTL).pt).rename(columns=inc)
        #         etasum = DataFrame((pTL + mTL).eta).rename(columns=inc)
        #         for frame in [msum, ptsum, etasum]:
        #             frame.index = mmuons.pt.index
        #         bgevv[i].mmsum = msum[msum > 12].dropna()
        #         bgevv[i].mptsum = ptsum.dropna()
        #         bgevv[i].metasum = etasum.dropna()
        #         bgevv[i].mqmpt = DataFrame(mmuons.pt.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqppt = DataFrame(pmuons.pt.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqmeta = DataFrame(mmuons.eta.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqpeta = DataFrame(pmuons.eta.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqmip3d = DataFrame(mmuons.ip3d.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqpip3d = DataFrame(pmuons.ip3d.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqmsip3d = DataFrame(mmuons.sip3d.sum(axis=1)).rename(columns=inc)
        #         bgevv[i].mqpsip3d = DataFrame(pmuons.sip3d.sum(axis=1)).rename(columns=inc)
                
        #         bev[i].sync()
        #         del pmuons, mmuons, msum, pTL, mTL
        
        #############################
        # Secondary Vertex Analysis #
        #############################
        ## Compute jet-vertex dR
        # sjvdr = computedR(sigjets,sigsv,['jet','sv'])
        # sjlist = []
        nsvframe = fastdR(sigjets,sigsv)#DataFrame()
        ## Loop over jets
        # for j in range(sigjets.eta.columns[-1]):
        #     ## Collect vertex dRs for each jet
        #     sjlist.append(sjvdr.filter(like=f"jet {j+1}"))
        #     sjlist[j] = sjlist[j].rename(columns=lambda x:int(x[9:11]))
        #     ## Remove dR >= 0.8, sum number of remaining vertices
        #     nsvframe[j+1] = np.sum(sjlist[j][sjlist[j]<0.8].fillna(0)/sjlist[j][sjlist[j]<0.8].fillna(0),axis=1)
        sigjets.nsv = nsvframe[nsvframe < 0.8].fillna(0)
        
        # if isLHE:
        for i in range(len(bgjets)):
            if bgjets[i].pt.size <= 0:
                bgjets[i].nsv = DataFrame()
                print(f"BG slice {i} had 0 passing events at nsv calc")
                continue
            # bjvdr = computedR(bgjets[i],bgsv[i],['jet','sv'])
            # bjlist = []
            nsvframe = fastdR(bgjets[i],bgsv[i])#DataFrame()
            # for j in range(bgjets[i].eta.columns[-1]):
            #     bjlist.append(bjvdr.filter(like=f"jet {j+1}"))
            #     bjlist[j] = bjlist[j].rename(columns=lambda x:int(x[9:11]))
            #     nsvframe[j+1] = np.sum(bjlist[j][bjlist[j]<0.8].fillna(0)/bjlist[j][bjlist[j]<0.8].fillna(0),axis=1)
            bgjets[i].nsv = nsvframe[nsvframe < 0.8].fillna(0)
            
        # else:
        #     bjvdr = computedR(bgjets,bgsv,['jet','sv'])
        #     bjlist = []
        #     nsvframe = DataFrame()
        #     for j in range(bgjets.eta.shape[1]):
        #         bjlist.append(bjvdr.filter(like=f"jet {j+1}"))
        #         bjlist[j] = bjlist[j].rename(columns=lambda x:int(x[9:11]))
        #         nsvframe[j+1] = np.sum(bjlist[j][bjlist[j]<0.8].fillna(0)/bjlist[j][bjlist[j]<0.8].fillna(0),axis=1)
        #     bgjets.nsv = nsvframe
        del nsvframe
            
        #######################################
        # Trigger Region Analysis & Weighting #
        #######################################

        for e in [ev] + bev:
            e.sync()
        
        for i in range(len(bgjets)):
            bgsj[i].trimto(bgjets[i].pt)
            if dataflag != -1:
                trigtensorcalc(bgjets[i],bgsj[i],bgevv[i],bgl1[i],bghlt[i])
            else: trigtensorcalc(bgjets[i],bgsj[i],bgevv[i],bgl1[i],bghlt[i],isdata=True)
            
        sigsj.trimto(sigjets.pt)              
        if dataflag != 1:
            trigtensorcalc(sigjets,sigsj,sigevv,sigl1,sighlt)
        else: trigtensorcalc(sigjets,sigsj,sigevv,sigl1,sighlt,isdata=True)
        
        for e in [ev] + bev:
            e.sync()
            
        ## HEM 15-16 veto
        if dataflag == True:
            sigevv.cut(~((sigevv.run > 319077) & (sigjets.eta < -1.17) & (sigjets.phi < -0.47) & (sigjets.phi > -1.97)))
            ev.sync()
        if dataflag != True:
            if sigevv.extweight.size:
                sigevv.extweight[(sigjets.eta < -1.17) & (sigjets.phi < -0.47) & (sigjets.phi > -1.97) &
                    sighlt.AK8PFJet500 & (sigevv.trg != "CX")] *= 21.09 / 59.8279
                sigevv.extweight[(sigjets.eta < -1.17) & (sigjets.phi < -0.47) & (sigjets.phi > -1.97) &
                    ~(sighlt.AK8PFJet500 & (sigevv.trg != "CX"))] *= 15.80 / 54.5365
        if dataflag != -1:
            for i in range(len(bgevv)):
                if bgevv[i].extweight.size:
                    bgevv[i].extweight[(bgjets[i].eta < -1.17) & (bgjets[i].phi < -0.47) & (bgjets[i].phi > -1.97) &
                        bghlt[i].AK8PFJet500 & (bgevv[i].trg != "CX")] *= 21.09 / 59.8279
                    bgevv[i].extweight[(bgjets[i].eta < -1.17) & (bgjets[i].phi < -0.47) & (bgjets[i].phi > -1.97) &
                        ~(bghlt[i].AK8PFJet500 & (bgevv[i].trg != "CX"))] *= 15.80 / 54.5365
        
        for e in [ev] + bev:
            e.sync()
            
        ## Luminosity + PU weighting
        if dataflag != 1 and sigevv.extweight.size:
            print(f"Dataflag: {dataflag}")
            sigevv.extweight[sighlt.AK8PFJet500 & (sigevv.trg != "CX")] *= 59.8279
            sigevv.extweight[~(sighlt.AK8PFJet500 & (sigevv.trg != "CX"))] *= 54.5365
            pucalc(sigevv,sighlt)
            
        if dataflag != -1:
            for i in range(len(bgevv)):
                if bghlt[i].AK8PFJet500.size:
                    bgevv[i].extweight[bghlt[i].AK8PFJet500 & (bgevv[i].trg != "CX")] *= 59.8279
                    bgevv[i].extweight[~(bghlt[i].AK8PFJet500 & (bgevv[i].trg != "CX"))] *= 54.5365
                    pucalc(bgevv[i],bghlt[i])
                
        ## PU weighting
        
        ##################################
        # Preparing Neural Net Variables #
        ##################################
        # if LOADMODEL and (not passplots):
            # bgjetframe['extweight'] = lumipucalc(bgjetframe)
            # bgrawframe['extweight'] = lumipucalc(bgrawframe)

        
        bgjetframe = DataFrame()
        extvars = ['event','extweight','npvs','npvsG']
        if LOADMODEL and False:#(not passplots):
            muvars = ['mpt','meta','mip']
        else: muvars = []#'mptsum','metasum','mmsum','mqmpt','mqppt','mqmeta','mqpeta','mqmip3d','mqpip3d',
                        #'mqmsip3d','mqpsip3d']
        if isLHE:
            bgpieces = []
            wtpieces = []
            
            for i in range(nlhe):
                tempframe = DataFrame()
                twgtframe = DataFrame()
                for prop in netvars + ['phi']:
                    twgtframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                for prop in extvars + muvars:
                    twgtframe[prop] = bgevv[i][prop].max(axis=1)
                if 'mmsum' in muvars:
                    twgtframe['MuSumvJetpT'] = np.divide(twgtframe['mmsum'],twgtframe['pt'])
                    twgtframe['MuQmvJetpT'] = np.divide(twgtframe['mqmpt'],twgtframe['pt'])
                    twgtframe['MuQpvJetpT'] = np.divide(twgtframe['mqppt'],twgtframe['pt'])
                    if i == nlhe-1:
                        muvars.append('MuSumvJetpT')
                        muvars.append('MuQmvJetpT')
                        muvars.append('MuQpvJetpT')
                ## DEBUG
                # if 'eta' in netvars:
                #     twgtframe['eta'] = abs(twgtframe['eta'])
                ## Add section for muon variables
                if LOADMODEL and False:#(not passplots):
                    for prop in ['pt','eta','ip']:
                        twgtframe[f"m{prop}"] = bmuons[i][prop][bmuons[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                if np.max(lheweights) > 1.0:
                    tempframe = twgtframe.sample(frac=lheweights[i],random_state=6,replace=True)
                else:
                    tempframe = twgtframe.sample(frac=lheweights[i],random_state=6)
                # twgtframe['extweight'] = twgtframe['extweight'] * lheweights[i]
                bgpieces.append(tempframe)
                #pickle.dump(tempframe, open(filefix+str(i)+"piece.p", "wb"))
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(bgpieces,ignore_index=True)
            bgrawframe = pd.concat(wtpieces,ignore_index=True)
            bgjetframe = bgjetframe.dropna()
            bgrawframe = bgrawframe.dropna()
            # if LOADMODEL and (not passplots):
            #     # bgjetframe['extweight'] = lumipucalc(bgjetframe)
            #     # bgrawframe['extweight'] = lumipucalc(bgrawframe)
            #     bgjetframe['extweight'] *= 54.54
            #     bgrawframe['extweight'] *= 54.54
            bgjetframe['val'] = 0
            bgrawframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]

        else:
            for prop in netvars + ['phi']:
                bgjetframe[prop] = bgjets[0][prop][bgjets[0]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            for prop in extvars + muvars:
                bgjetframe[prop] = bgevv[0][prop][1]
            if 'mmsum' in muvars:
                    bgjetframe['MuSumvJetpT'] = np.divide(bgjetframe['mmsum'],bgjetframe['pt'])
                    muvars.append('MuSumvJetpT')
                    bgjetframe['MuQmvJetpT'] = np.divide(bgjetframe['mqmpt'],bgjetframe['pt'])
                    muvars.append('MuQmvJetpT')
                    bgjetframe['MuQpvJetpT'] = np.divide(bgjetframe['mqppt'],bgjetframe['pt'])
                    muvars.append('MuQpvJetpT')
            ## DEBUG
            # if 'eta' in netvars:
            #     bgjetframe['eta'] = abs(bgjetframe['eta'])
            ## Add section for muon variables
            # if LOADMODEL and (not passplots):
            #     # for prop in ['pt','eta','ip']:
            #     #     bgjetframe[f"m{prop}"] = bmuons[0][prop][bmuons[0]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            
            #     if dataflag != -1:
            #         # bgjetframe['extweight'] = lumipucalc(bgjetframe)
            #         bgjetframe['extweight'] *= 54.54
            bgjetframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        
        #nbg = bgtrnframe.shape[0]
            
        sigjetframe = DataFrame()
        for prop in netvars + ['phi']:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        for prop in extvars + muvars:
            sigjetframe[prop] = sigevv[prop]
        if 'mmsum' in muvars:
            sigjetframe['MuSumvJetpT'] = np.divide(sigjetframe['mmsum'],sigjetframe['pt'])
            sigjetframe['MuQmvJetpT'] = np.divide(sigjetframe['mqmpt'],sigjetframe['pt'])
            sigjetframe['MuQpvJetpT'] = np.divide(sigjetframe['mqppt'],sigjetframe['pt'])
        ## DEBUG
        # if 'eta' in netvars:    
        #     sigjetframe['eta'] = abs(sigjetframe['eta'])   
        ## Signal MC specific pt reweighting
        if dataflag != 1:
            ptwgt = 3.9 - (0.4*np.log2(sigjetframe.pt))
            ptwgt[ptwgt < 0.1] = 0.1
            sigjetframe.extweight = sigjetframe.extweight * ptwgt
        ## Add section for muon variables
        # if LOADMODEL and (not passplots):
        #     # for prop in ['pt','eta','ip']:
        #     #     sigjetframe[f"m{prop}"] = muons[prop][muons['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
    
        #     if dataflag != True:
        #         # sigjetframe['extweight'] = lumipucalc(sigjetframe)
        #         sigjetframe['extweight'] *= 54.54
        # sigjetframe['extweight'] = sigjetframe['extweight'] * ic.sigweight[fnum]
        sigjetframe['val'] = 1
        sigtrnframe = sigjetframe[sigjetframe['event']%2 == 0]
        #nsig = sigtrnframe.shape[0]
        
        
        print(f"{Skey} cut to {sigjetframe.shape[0]} events")
        print(f"{Bkey} has {bgjetframe.shape[0]} intended events")
        
        bgjetframe = bgjetframe.reset_index(drop=True)
        sigjetframe = sigjetframe.reset_index(drop=True)
        
        extvars = extvars + muvars + ['val'] + ['phi']
        
#        ##
#        
#        if bgtrnframe.shape[0] < sigtrnframe.shape[0]:
#            bgtrnlst = []
#            for i in range(int(np.floor(sigtrnframe.shape[0] / bgtrnframe.shape[0]))):
#                bgtrnlst.append(bgtrnframe)
#            bgtrnframe = pd.concat(bgtrnlst,ignore_index=True)
#                
#        
#        ##
        #######################
        # Training Neural Net #
        #######################
        
            
        #if not isLHE:
        #    X_test = pd.concat([bgjetframe.drop(bgtrnframe.index), sigjetframe.drop(sigtrnframe.index)])#,ignore_index=True)
        #    X_train = pd.concat([bgtrnframe,sigtrnframe])#,ignore_index=True)
        #    passnum = 0.9       
        #else:
        if LOADMODEL and not TUTOR:
            if isLHE:
                bgjetframe=bgrawframe
            ## Normalize event number between QCD and data samples
            if dataflag == True:
                bgjetframe['extweight'] = bgjetframe['extweight'] * np.sum(sigjetframe['extweight'])/np.sum(bgjetframe['extweight'])
                
            X_inputs = pd.concat([bgjetframe,sigjetframe])
            W_inputs = X_inputs['extweight']
#            W_inputs = lumipucalc(X_inputs)
            Y_inputs = X_inputs['val']
            X_inputs = X_inputs.drop(extvars,axis=1)
            model = keras.models.load_model('archive/weighted.hdf5', compile=False) #archive
            scaler = pickle.load( open("archive/weightedscaler.p", "rb" ) )
            ##
            #print(scaler.transform(bgpieces[1].drop('val',axis=1)))
            ##
            X_inputs = scaler.transform(X_inputs)
        else:
            X_test = pd.concat([bgjetframe.drop(bgtrnframe.index),sigjetframe.drop(sigtrnframe.index)])
             ##
            if bgtrnframe.shape[0] < sigtrnframe.shape[0]:
                bgtrnlst = []
                for i in range(int(np.floor(sigtrnframe.shape[0] / bgtrnframe.shape[0]))):
                    bgtrnlst.append(bgtrnframe)
                bgtrnframe = pd.concat(bgtrnlst,ignore_index=True)
                print (f"bg now {bgtrnframe.shape[0]} to sg {sigtrnframe.shape[0]}")
            ##

            X_train = pd.concat([bgtrnframe,sigtrnframe])

            
            #for plot in plots:
            #plots[plot].title = 'Post-Weighted Training'
            W_test = X_test['extweight']
            W_train = X_train['extweight']
#            W_test = lumipucalc(X_test)
#            W_train = lumipucalc(X_train)
                
            Y_test = X_test['val']
            Y_train= X_train['val']
        
            X_test = X_test.drop(extvars,axis=1)
            X_train = X_train.drop(extvars,axis=1)
            
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if TUTOR == True:
                tutor(X_train,X_test,Y_train,Y_test)
                sys.exit()
            else: history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True,verbose=False)

            rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
            trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train).ravel())
            test_loss, test_acc = model.evaluate(X_test, Y_test)
            print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
    
        passnum = 0.8
        ##################################
        # Analyzing and Plotting Outputs #
        ##################################
        

        
        if LOADMODEL:
            diststt = model.predict(X_inputs[Y_inputs==1])
            distbtt  = model.predict(X_inputs[Y_inputs==0])
        else:
            diststr = model.predict(X_train[Y_train==1])
            distste = model.predict(X_test[Y_test==1])
            distbtr = model.predict(X_train[Y_train==0])
            distbte = model.predict(X_test[Y_test==0])
            diststt = model.predict(scaler.transform(sigjetframe.drop(extvars,axis=1)))
            distbtt = model.predict(scaler.transform(bgjetframe.drop(extvars,axis=1)))
        
        #if isLHE:
        #    for i in range(nlhe):
        #        piece = wtpieces[i].drop(extvars,axis=1)
        #        piece = piece.reset_index(drop=True)
        #        piece = scaler.transform(piece)
        #        lhedist = model.predict(piece)
        #        #if POSTWEIGHT:
        #        lheplots['dist'+str(i)].fill(lhedist,wtpieces[i]['extweight'])
        #        #else:
        #        #    lheplots['dist'+str(i)].fill(lhedist)
        
        if LOADMODEL:
            plots['DistSte'].fill(diststt,W_inputs[Y_inputs==1])
            plots['DistBte'].fill(distbtt,W_inputs[Y_inputs==0])
            
            ## Store the output confidence values and their weights
            distsf = DataFrame(diststt)
            distsf['W'] = W_inputs[Y_inputs==1].reset_index(drop=True)
            ## Sort both of them together by the confidence values
            distsf = distsf.sort_values(by=[0])
            ## Store the cumulative sum of the weights
            distsf['W_csum'] = distsf.W.cumsum()
            ## Decide what weighted interval each volume should encompass
            isize = distsf.W_csum.max() / 10
            ## label which of those intervals each event falls into
            distsf['bin'] = (distsf.W_csum / isize).dropna().apply(math.floor)
            
            for i in range(1,10):
                plots['SensS'][1][i] = distsf[distsf['bin']==i-1][0].max()
                plots['SensB'][1][i] = plots['SensS'][1][i]
            plots['SensS'].fill(diststt,W_inputs[Y_inputs==1])
            plots["SensB"].fill(distbtt,W_inputs[Y_inputs==0])

        elif not passplots:
            hist = DataFrame(history.history)
            #for h in history:
                #hist = pd.concat([hist,DataFrame(h.history)],ignore_index=True)
            hist['epoch'] = history.epoch
            plots['LossvEpoch'][0]=hist['loss']
            if 'acc' in hist.columns:
                plots['AccvEpoch'][0]=hist['acc']
            elif 'accuracy' in hist.columns:
                plots['AccvEpoch'][0]=hist['accuracy']
            else:
                plots['AccvEpoch'][0]=hist['loss']
                #plots['LossvEpoch'][0]=hist['epoch']
                #plots['AccvEpoch'][0]=hist['epoch']
            plots['LossvEpoch'].plot()
            plots['AccvEpoch'].plot()
                
            plots['DistStr'].fill(diststr,W_train[Y_train==1])
            plots['DistSte'].fill(distste,W_test[Y_test==1])
            plots['DistBtr'].fill(distbtr,W_train[Y_train==0])
            plots['DistBte'].fill(distbte,W_test[Y_test==0])
            ## Blinding code for data vs QCD plots

            
            plt.clf()
            plt.plot([0,1],[0,1],'k--')
            plt.plot(rocx,rocy,'red')
            plt.plot(trocx,trocy,'b:')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(['y=x','Validation','Training'])
            plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
            plt.savefig(plots['Distribution'].fname+'ROC')
            
        #else:
        #    plots['DistStr'].fill(diststr)
        #    plots['DistSte'].fill(distste)
        #    plots['DistBtr'].fill(distbtr)
        #    plots['DistBte'].fill(distbte)
        #    plt.clf()
            
        for col in netvars + muvars + ['npvs','npvsG','phi']:
            if not passplots:
                vplots['BG'+col].fill(bgjetframe[col],bgjetframe['extweight'])
                vplots['SG'+col].fill(sigjetframe[col],sigjetframe['extweight'])
            else:
                pplots['SG'+col].fill(sigjetframe[col],sigjetframe['extweight'])
                pplots['SPS'+col].fill(sigjetframe[diststt > passnum][col],sigjetframe[diststt > passnum]['extweight'])
                pplots['SFL'+col].fill(sigjetframe[diststt <= passnum][col],sigjetframe[diststt <= passnum]['extweight'])
                pplots['BG'+col].fill(bgjetframe[col],bgjetframe['extweight'])
                pplots['BPS'+col].fill(bgjetframe[distbtt > passnum][col],bgjetframe[distbtt > passnum]['extweight'])
                pplots['BFL'+col].fill(bgjetframe[distbtt <= passnum][col],bgjetframe[distbtt <= passnum]['extweight'])

    # if False:#LOADMODEL:
    #     #if gROOT.FindObject('Combined.root'):
    #      #   rfile = TFile('Combined.root','UPDATE')
    #     rfile = TFile('Combined.root','UPDATE')
    #     if dataflag:
    #         th1 = plots['DistSte'].toTH1('data_obs')
    #         th12 = plots['DistBte'].toTH1('DnetQCD')
    #     else:
    #         th1 = plots['DistSte'].toTH1('SnetSMC')
    #         th12 = plots['DistBte'].toTH1('SnetQCD')
    #     rfile.Write()
    #     rfile.Close()

#    if dataflag == False:
#        outfile = uproot.recreate("Combined.root")
#        outfile["SnetQCD"] = plots['DistBte'].toTH1()
#        outfile["SnetQCDerr"] = plots['DistBte'].errToTH1()
#        outfile["SnetSMC"] = plots['DistSte'].toTH1()
#        outfile["SnetSMCerr"] = plots['DistSte'].errToTH1()
#    elif dataflag == True:
#        outfile = uproot.recreate("CombinedGen.root")
#        outfile["data_obs"] = plots['DistSte'].toTH1()
#        outfile["data_obserr"] = plots['DistSte'].errToTH1()
#        outfile["DnetQCD"] = plots['DistBte'].toTH1(scale=0.5)
#        outfile["DnetQCDerr"] = plots['DistBte'].errToTH1(scale=0.5)
        

    for p in [plots['DistStr'],plots['DistSte'],plots['DistBtr'],plots['DistBte'],plots['SensB'],plots['SensS']]:
        if sum(p[0]) != 0:
            p.ival = sum(p[0])
            p.ndivide(sum(p[0]))
        
    if False:#dataflag == True:
        for i in range(len(plots['DistSte'][0])):
            if plots['DistSte'][1][i] >= passnum:
                plots['DistSte'][0][i] = 0
                plots['DistSte'].ser[i] = 0
        
    if LOADMODEL:
        leg = [Sname,Bname]
        Sens = np.sqrt(np.sum(np.power(plots['SensS'][0],2)/plots['SensB'][0]))
        print(f"Calculated Senstivity of {Sens}")
            
        plt.clf()
        pickle.dump(plots, open('debug.p','wb'))
        plots['DistSte'].make(linestyle='-',**plargs[Skey])
        plots['DistBte'].make(linestyle=':',**plargs[Bkey])
        plots['Distribution'].plot(same=True,legend=leg)
        plt.clf()
        plots['DistSte'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['DistBte'].make(linestyle=':',logv=True,**plargs[Bkey])
        plots['DistributionL'].plot(same=True,logv=True,legend=leg)
        plots['SensS'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['SensB'].xlabel = f"Confidence (S={Sens})"
        plots['SensB'].plot(legend=leg,same=True,linestyle=':',logv=True,**plargs[Bkey])
    elif not passplots:
        leg = ['Signal (training)','Background (training)','Signal (testing)', 'Background(testing)']
            
        plt.clf()
        plots['DistStr'].make(linestyle='-',**plargs[Skey])
        plots['DistBtr'].make(linestyle='-',**plargs[Bkey])
        plots['DistSte'].make(linestyle=':',**plargs[Skey])
        plots['DistBte'].make(linestyle=':',**plargs[Bkey])
        plots['Distribution'].plot(same=True,legend=leg)
    
        plt.clf()
        plots['DistStr'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['DistBtr'].make(linestyle='-',logv=True,**plargs[Bkey])
        plots['DistSte'].make(linestyle=':',logv=True,**plargs[Skey])
        plots['DistBte'].make(linestyle=':',logv=True,**plargs[Bkey])
        plots['DistributionL'].plot(same=True,logv=True,legend=leg)
    

    
    if passplots:
        if dataflag != True:
            for col in netvars:
                for plot in pplots:
                    pplots[plot].ndivide(sum(abs(pplots[plot][0]+.0001)))
                    
        for col in netvars + muvars + ['npvs','npvsG','phi']:   
            plt.clf()
            pplots['SG'+col].make(color='red'  ,linestyle='-',htype='step')
            pplots['SFL'+col].make(color='black',linestyle='--',htype='step')
            pplots['SPS'+col].make(color='blue' ,linestyle=':',htype='step')
            pplots[col].plot(same=True,legend=[f"All {Sname}", f"Failing {Sname}",f"Passing {Sname}"])
            
            plt.clf()
            pplots['BG'+col].make(color='red'  ,linestyle='-',htype='step')
            pplots['BFL'+col].make(color='black',linestyle='--',htype='step')
            pplots['BPS'+col].make(color='blue' ,linestyle=':',htype='step')
            pplots['B'+col].plot(same=True,legend=[f"All {Bname}",f"Failing {Bname}",f"Passing {Bname}"])
    else:
        if False:#dataflag != True:
            for col in netvars:
                for plot in vplots:
                    vplots[plot].ndivide(sum(abs(vplots[plot][0]+.0001)))
        for col in netvars + muvars + ['npvs','npvsG']:
            plt.clf()
            vplots['SG'+col].make(linestyle='-',**plargs[Skey])
            vplots['BG'+col].make(linestyle=':',**plargs[Bkey],error=dataflag)
            vplots[col].plot(same=True,legend=[Sname,Bname])
        
    #if POSTWEIGHT:
    #model.save('postweighted.hdf5')
    #pickle.dump(scaler, open("postweightedscaler.p", "wb"))
    #else:
    
    # pickle.dump(sigevv,open('sigevv.p','wb'))
    # pickle.dump(sigjets,open('sigjets.p','wb'))
    # pickle.dump(sigsj,open('sigsj.p','wb'))
    
    # pickle.dump(bgevv,open('bgevv.p','wb'))
    # pickle.dump(bgjets,open('bgjets.p','wb'))
    # pickle.dump(bgsj,open('bgsj.p','wb'))
        
    if not LOADMODEL:
        model.save('weighted.hdf5')
        pickle.dump(scaler, open("weightedscaler.p", "wb"))
        
    elif not passplots:
        arcdict = {"plots":plots,"vplots":vplots}
        if ic.signame:
            if ic.bgname:
                setname = ic.signame+' vs '+ic.bgname
            else:
                setname = ic.signame
        elif ic.bgname:
            setname = ic.bgname
        else: setname = ''
        arcdict.update({'setname':setname})
        if setname:
            pickle.dump(arcdict, open(plots['Distribution'].fname.split('/')[0]+'/'+setname+'.p', "wb"))
        else: pickle.dump(arcdict, open(plots['Distribution'].fname.split('/')[0]+'/arcdict.p', "wb"))
    # pickle.dump(bgjetframe, open("bgjet.p","wb"))

    
        
    #%%
    #return auc(rocx,rocy)
        #sys.exit()

## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        nrgs = len(sys.argv)
        sigfile, bgfile = '',''
        passplots=False
        LOADMODEL = True
        TUTOR = False
        ## Check for file sources
        for i in range(nrgs):
            arg = sys.argv[i]
            if '-s' in arg:
                sigfile = sys.argv[i+1]
            elif '-b' in arg:
                bgfile = sys.argv[i+1]
            elif ('-Pass' in arg) or ('-pass' in arg):
                passplots = True
            elif ('-Train' in arg) or ('-train' in arg):
                LOADMODEL=False
            elif ('-Tutor' in arg) or ('-tutor' in arg):
                TUTOR = True
                LOADMODEL=False
            #else: dialogue()
        #print('-')
        #print('sigfiles',sigfiles,'datafiles',datafiles)
        #print('-')
        if not sigfile or not bgfile:
            dialogue()
        else:
            ana(sigfile,bgfile,LOADMODEL,TUTOR,passplots)

    else:
        dialogue()
        
def dialogue():
    print("Expected:\n mndwrm.py [-LHE] <-f/-l>s (signal.root) <-f/-l>b (background.root)")
    print("---formatting flags--")
    print("-s     Marks the signal .json file")
    print("-b     Marks the background .json file")
    print("-Pass  Indicates pass/fail plots should be made instead of regular plots")
    print("-Train Indicates a new neural network should be trained")
    print("-Tutor Indicates the neural network hyperparameters should be optimized")
    sys.exit(0)
    
if __name__ == "__main__":
    main()

#%%