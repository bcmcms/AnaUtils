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
import matplotlib.patheffects as PathEffects
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
# from uproot_methods import TLorentzVectorArray as TLVA
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform as squareform
import re, pdb

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
##Toggles experimental subnet training code
subnet = False
REGION = ''#'lepton','lowmass'
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
    pvals = uproot.open('archive/PU_ratio_2021_05_26.root').get('PU_ratio').values
    fvals = uproot.open('archive/PU_ratio_2021_05_26.root').get('PU_ratio_HLT_AK8PFJet330').values
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
    ev.metpt = DataFrame(events.array('MET_pt')).rename(columns=inc)
    if 'Generator_weight' in events:
        ev.genweight = DataFrame(events.array('Generator_weight')).rename(columns=inc)
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
        pdgframe = np.logical_and(np.abs(pdgid) == 5, pdgpt > 15)
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

def senscalc(sigv, bgv, sigw):
    sensf = DataFrame(sigv)
    sensf['W'] = sigw
    sensf = sensf.sort_values(by=[0])
    sensf['W_csum'] = sensf.W.cumsum()
    isize = sensf.W_csum.max()/10
    sensf['bin'] = (sensf.W_csum / isize).dropna().apply(math.floor)
    edges = [-np.inf]
    for i in range(1,10):
        edges.append(sensf[sensf['bin'] == i-1][0].max())
    return edges

def distance_corr(var_1, var_2, normedweight, power=1):
    """
    https://github.com/gkasieczka/DisCo
    var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    # import pdb; pdb.set_trace()
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])

    yy = tf.transpose(xx)
    amat = tf.abs(xx-yy)

    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])

    yy = tf.transpose(xx)
    bmat = tf.abs(xx-yy)

    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)

    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)
    # Amat = K.clip(Amat,K.epsilon(),1)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)
    # Bmat = K.clip(Bmat,K.epsilon(),1)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)

    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
    return dCorr

def disco_focal_loss(alpha=.25, gamma=2.,lamb=.1):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def disco_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # import pdb; pdb.set_trace()
        target = tf.reshape(y_pred[:,   1:2], [-1,1])
        y_pred = tf.reshape(y_pred[:,   :1], [-1,1])
        # y_pred = K.clip(y_pred,K.epsilon(),1)
        # target = K.clip(target,K.epsilon(),1)
        normedweight = tf.ones_like(target)


        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        FLout = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        DCout = distance_corr(target,y_pred,normedweight,1)
        print("Debug:")
        tf.print(FLout, " - ", lamb*DCout, output_stream=sys.stdout)
        print("-----")
        try:
            tf.debugging.assert_all_finite(DCout,'fuck')
        except: pdb.set_trace()


        return FLout + lamb*DCout

    return disco_focal_loss_fixed

def submodel(ndim):
    layerin = keras.layers.Input(shape=(ndim,),name='layerin')
    layer = keras.layers.Dense(8, activation=tf.nn.relu) (layerin)
    layer = keras.layers.Dense(8, activation=tf.nn.relu) (layer)
    layer = keras.layers.Dense(8, activation=tf.nn.relu) (layer)
    layerout = keras.layers.Dense(1, activation=tf.nn.sigmoid) (layer)

    lossin = keras.layers.Input(shape=(1,),name='lossin')
    out = keras.layers.concatenate([layerout,lossin], name='out')

    model = keras.models.Model(inputs=[layerin,lossin],outputs=out,name='model')
    return model

#%%

def ana(sigfile,bgfile,LOADMODEL=True,TUTOR=False,passplots=False,NET="F",subf=''):
    #%%
    cplot = Hist(10,(0,10),'stages','events','cutplot')
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
    if NET == "F": netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','n2b1','submass1','submass2','subtau1','subtau2','nsv']
    elif NET == "A": netvars = ['pt','eta','CSVV2','DeepB','nsv']
    elif NET == "B": netvars = ['H4qvs','n2b1','submass1','subtau1']
    elif NET == "C":netvars = ['mass','msoft','DDBvL','submass2','subtau2']
    else: raise NameError("Invalid NET specified")
    l1vars = ['SingleJet180','Mu7_EG23er2p5','Mu7_LooseIsoEG20er2p5','Mu20_EG10er2p5','SingleMu22',
              'SingleMu25','DoubleJet112er2p3_dEta_Max1p6','DoubleJet150er2p5']
    hltvars = ['AK8PFJet500','Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ','Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
               'Mu27_Ele37_CaloIdL_MW','Mu37_Ele27_CaloIdL_MW',
               'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
               'DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71']
    slimvars = ['pt','eta','phi','btagDeepB','btagDeepFlavB','puId']

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
    if subnet or NET == "B":
        lamb = 0.01
        mname = "B"
        sname = "B"
        disconame = 'C'
        model = submodel(5)
        optimizer  = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer,
                  loss=[disco_focal_loss(alpha, gamma, lamb)],
                  metrics=['accuracy'])
    else:
        if NET == "F":
            mname = "weighted"
            sname = "weightedscaler"
        else:
            mname = NET
            sname = NET
        disconame = 'C'
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
        "mass":     Hist(35 ,(60,200)   ,'mass for highest pT jet','Fractional Distribution','netplots/pmass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet in all (red), passing signal (blue), and signal (black) events','Fractional Distribution','netplots/pCSVV2'),
        "DeepB":    Hist(14 ,(0.35,1.05),'DeepB for highest pT jet','Fractional Distribution','netplots/pDeepB'),
        "msoft":    Hist(35 ,(60,200)   ,'msoft for highest pT jet in all (red), passing (blue), and failing  (black) events','Fractional Distribution','netplots/pmsoft'),
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
        "eta":      Hist(30 ,(-3,3)     ,'|eta| for highest pT jet','Fractional Distribution','netplots/eta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/phi'),
        "mass":     Hist(35 ,(60,200)   ,'mass for highest pT jet','Fractional Distribution','netplots/mass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet','Fractional Distribution','netplots/CSVV2'),
        "DeepB":    Hist(14 ,(0.35,1.05),'DeepB for highest pT jet','Fractional Distribution','netplots/DeepB'),
        "msoft":    Hist(35 ,(60,200)   ,'msoft for highest pT jet','Fractional Distribution','netplots/msoft'),
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
        'nlep':     Hist(5 ,(-0.5,4.5)     ,'# of leptons passing cuts','Distribution','netplots/nlep'),
        "metpt":    Hist(29 ,(30,300)   ,'MET pT','Fractional Distribution','netplots/metpt'),

    }
    lepplots = {
        "mpt":      Hist(50,(0,500)     ,'pT for highest pT muon','Distribution','netplots/mupt'),
        "meta":     Hist(6 ,(0,3)       ,'|eta| for highest pT muon','Distribution','netplots/mueta'),
        "mminiPFRelIso_all":    Hist(10 ,(0,0.2)    ,'miniIso for highest pT muon','Distribution','netplots/mumiso'),
        'msip3d':   Hist(40 ,(0,4)      ,'sip3d for highest pT muon','Distribution','netplots/musip3d'),
        'mjetdr':   Hist(10 ,(0.8,4.8)  ,'candidate fatjet dR for highest pT muon','Distribution','netplots/mujetdr'),
        "ept":      Hist(50,(0,500)     ,'pT for highest pT electron','Distribution','netplots/elpt'),
        "eeta":     Hist(6 ,(0,3)       ,'|eta| for highest pT electron','Distribution','netplots/eleta'),
        "eminiPFRelIso_all":    Hist(20 ,(0,0.1)    ,'miniIso for highest pT electron','Distribution','netplots/elmiso'),
        'esip3d':   Hist(40 ,(0,4)      ,'sip3d for highest pT electron','Distribution','netplots/elsip3d'),
        'ejetdr':   Hist(10 ,(0.8,4.8)  ,'candidate fatjet dR for highest pT electron','Distribution','netplots/eljetdr'),

        }
    vplots.update(lepplots)
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
            bgevents = [bgf.get('Events')]
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
                bgjets, bgevv = loadjets(PhysObj('bgjets'),PhysObj("bgev"),bgevents[0])
            else:
                bgjets, bgevv = loadjets(PhysObj('bgjets'),PhysObj("bgev"),bgevents[0],True)
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

            # slimjets = PhysObj('slimjets')
            # slimjets.eta= DataFrame(sigevents.array('Jet_eta', executor=executor)).rename(columns=inc)
            # slimjets.phi= DataFrame(sigevents.array('Jet_phi', executor=executor)).rename(columns=inc)
            # slimjets.pt = DataFrame(sigevents.array('Jet_pt' , executor=executor)).rename(columns=inc)
            # slimjets.mass=DataFrame(sigevents.array('Jet_mass', executor=executor)).rename(columns=inc)
            # slimjets.DeepB = DataFrame(sigevents.array('Jet_btagDeepB', executor=executor)).rename(columns=inc)
            # slimjets.DeepFB= DataFrame(sigevents.array('Jet_btagDeepFlavB', executor=executor)).rename(columns=inc)
            # slimjets.puid = DataFrame(sigevents.array('Jet_puId', executor=executor)).rename(columns=inc)


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

            ev = Event(bs,sigjets,As,higgs,sigsv,sigevv,sigl1,sighlt)
        else:
            if dataflag != 1:
                Hpt = DataFrame(sigevents.array('GenPart_pt'))[DataFrame(sigevents.array('GenPart_pdgId'))==25][DataFrame(sigevents.array('GenPart_status'))==62].max(axis=1)
                ptwgt = 3.9 - (0.4*np.log2(Hpt))
                ptwgt[ptwgt < 0.1] = 0.1
                sigevv.extweight[1] *= ptwgt
                
            if 'TTbar' in ic.bgname:
                for i in range(len(bgevents)):
                    pdgida  = bgevents[i].array('GenPart_pdgId')
                    pdgstat = bgevents[i].array('GenPart_status')
                    tpt = DataFrame(bgevents[0].array('GenPart_pt')[pdgida==6][pdgstat[pdgida==6]==62]).rename(columns=inc)
                    tbpt = DataFrame(bgevents[0].array('GenPart_pt')[pdgida==-6][pdgstat[pdgida==-6]==62]).rename(columns=inc)
                    
                    twgt = ((0.103*np.exp(-0.0118*tpt[1])) - (0.000134*tpt[1]) + 0.973)**0.5
                    twgt *= ((0.103*np.exp(-0.0118*tbpt[1])) - (0.000134*tbpt[1]) + 0.973)**0.5
                    twgt = 1 + 0.5*(twgt - 1)
                    
                    bgevv[i].extweight[1] *= twgt
            ev = Event(sigjets,sigsv,sigevv,sigl1,sighlt)


        bev = []
        for i in range(len(bgjets)):
            bev.append(Event(bgjets[i],bgsv[i],bgevv[i],bgl1[i],bghlt[i]))

        # import pdb; pdb.set_trace()
    
        for jets in bgjets+[sigjets]:
            ##['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','n2b1','submass1','submass2','subtau1','subtau2','nsv']
            
            jets.cut(jets.pt > 170)
            jets.cut(abs(jets.eta)<2.4)
            jets.cut(jets.DDBvL > 0.8)
            jets.cut(jets.DeepB > 0.4184)
            if 'lowmass' in REGION:
                jets.cut(jets.msoft > 70)
                jets.cut(jets.mass > 70)
                jets.cut(jets.mass < 90)
                jets.cut(jets.msoft < 90)  
            else:
                jets.cut(jets.msoft > 90)
                jets.cut(jets.mass > 90)
                jets.cut(jets.mass < 200)
                jets.cut(jets.msoft < 200)  
        cplot[0][0] = sigjets.pt.shape[0]
        for evv in bgevv + [sigevv]:
            evv.cut(evv.npvsG >= 1)
            if 'lepton' in REGION:
                evv.cut(evv.metpt > 30)
                
        for i in range(len(bgevv)):
            if ic.bgqcd[i] == -1:
                bgevv[i].cut(bgevv[i].nGprt == 0)
            elif ic.bgqcd[i]:
                bgevv[i].cut(bgevv[i].nGprt >= 1)
                
        ev.sync()
        cplot[0][1] = sigjets.pt.shape[0]
            
        ## Temporary mass rescaling for exclusion zone
        if 'lowmass' in REGION:
            for jets in bgjets + [sigjets]:
                #mass_scaled = mass*(1.3 + 0.536*(math.exp(pow((mass - 70.)/20., 3)) - 1))
                #msoft_scaled = msoft*(1.3 + 0.536*(math.exp(pow((msoft - 70.)/20., 0.8)) - 1))
                jets.mass  = jets.mass*(1.3+0.536*(np.exp(np.power((jets.mass - 70)/20,3)) -1 ))
                jets.msoft = jets.msoft*(1.3 + 0.536*(np.exp(np.power((jets.msoft - 70)/20, 0.8)) - 1))
                jets.mass += jets.mass*0.2
                jets.msoft += jets.mass*0.2

        if (not dataflag) and (not LOADMODEL):
            bs.cut(bs.pt>5)
            bs.cut(abs(bs.eta)<2.4)
            ev.sync()
            sigsj.cut(sigsj.btagDeepB > 0.1241)
            sigsj.cut(sigsj.btagDeepFlavB > 0.277)
            sigsj.cut(sigsj.puId > 0)
            sigsj.trimto(sigjets.eta)

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
        # if (LOADMODEL) and False:#(not passplots):
        #     muons = PhysObj('Muon',sigfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d')
        #     muons.ip = abs(muons.dxy / muons.dxyErr)
        #     muons.eta = abs(muons.eta)
        #     ev.register(muons)
        #     muons.cut(muons.softId > 0.9)
        #     muons.cut(abs(muons.eta) < 2.4)
        #     muons.cut(muons.pt > 7)
        #     muons.cut(muons.ip > 2)
        #     #muons.cut(muons.ip3d < 0.5)
        #     ev.sync()

        #     ## Apply these cuts to data events as well
        #     bmuons = []
        #     for i in range(len(bgjets)):
        #         idx = i#fnum*nlhe + i
        #         bmuons.append(PhysObj('Muon',bgfiles[idx],'softId','eta','pt','dxy','dxyErr','ip3d'))
        #         bmuons[i].eta = abs(bmuons[i].eta)
        #         bmuons[i].ip = abs(bmuons[i].dxy / bmuons[i].dxyErr)
        #         bev[i].register(bmuons[i])
        #         bmuons[i].cut(bmuons[i].softId > 0.9)
        #         bmuons[i].cut(abs(bmuons[i].eta) < 2.4)
        #         bmuons[i].cut(bmuons[i].pt > 7)
        #         bmuons[i].cut(bmuons[i].ip > 2)
        #         bmuons[i].cut(bmuons[i].ip3d < 0.5)
        #         bev[i].sync()

        cplot[0][2] = sigjets.pt.shape[0]

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

            fjets = blist[0]<0.8
            fjets = fjets.astype(int)
            for i in range(1,4):
                fjets = fjets + (blist[i]<0.8)
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
        sigjets.nsv = DataFrame()
        sigjets.nsv[1] = np.sum(nsvframe < 0.8,axis=1)


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
            bgjets[i].nsv = DataFrame()
            bgjets[i].nsv[1] = np.sum(nsvframe < 0.8,axis=1)

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

        # for jets in bgjets + [sigjets]:
        #     jets.nsv += jets.nsv*0.2

        #######################################
        # Trigger Region Analysis & Weighting #
        #######################################

        for e in [ev] + bev:
            e.sync()
            
        cplot[0][3] = sigjets.pt.shape[0]

        for i in range(len(bgjets)):
            if bgjets[i].pt.shape[0]:
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

        cplot[0][4] = sigjets.pt.shape[0]

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
            
        cplot[0][5] = sigjets.pt.shape[0]

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
        
        #########################
        # Muon-Region Selection #
        #########################

        # import pdb; pdb.set_trace()
        # # Special region-selection muon cut logic
        if (LOADMODEL) and (not passplots) and ('lepton' in REGION):
            muons = PhysObj('Muon',sigfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d','charge','mass','phi','sip3d','mediumId','miniPFRelIso_all','mvaTTH')
            muons.ip = abs(muons.dxy / muons.dxyErr)
            mframe = fastdR(sigjets, muons)
            mframe = mframe[muons.pt != 0].dropna(how='all')
            muons.jetdr = mframe
            muons.trimto(sigjets.pt)
            Event(muons).sync()

            bmuons = []
            for i in range(len(bgjets)):
                bmuons.append(PhysObj('Muon',bgfiles[i],'softId','eta','pt','dxy','dxyErr','ip3d','charge','mass','phi','sip3d','mediumId','miniPFRelIso_all','mvaTTH'))
                bmuons[i].ip = abs(bmuons[i].dxy / bmuons[i].dxyErr)
                mframe = fastdR(bgjets[i], bmuons[i])
                mframe = mframe[bmuons[i].pt != 0].dropna(how='all')
                bmuons[i].jetdr = mframe
                bmuons[i].trimto(bgjets[i].pt)
                Event(bmuons[i]).sync()
                
                

            # cutflow = Hist(7,(-0.5,6.5),'all/pt/eta/mediumId/miniPFRelIso_all/sip3d/jetdR','passing',f"{ic.signame}_mCutflow")
            for muon in bmuons + [muons]:
                muon.eta = abs(muon.eta)
                muon.cut(muon.pt > 30)
                muon.cut(abs(muon.eta) < 2.4)
                muon.cut(muon.mediumId > 0.9)
                muon.cut(muon.miniPFRelIso_all < 0.2)
                muon.cut(abs(muon.sip3d) < 4.0)
                muon.cut(muon.jetdr > 0.8)
                muon.cut(muon.mvaTTH > -0.4)
                ## Only keep the highest passing pT muon
                muon.cut(muon.pt.rank(axis=1,method='first',ascending=False) == 1)
                ## Muon crunch
                for prop in muon:
                    muon[prop] = DataFrame(muon[prop].max(axis=1)).rename(columns=inc)

            

            
            elecs = PhysObj('Electron',sigfiles[fnum],'pt','eta','phi','miniPFRelIso_all','mvaFall17V1Iso_WP80','sip3d','mvaTTH')
            eframe = fastdR(sigjets, elecs)
            elecs.jetdr = eframe[elecs.pt != 0].dropna(how='all')

            eledoc = sigevents.get("Electron_vidNestedWPBitmap").title.decode()
            elearr = sigevents.array("Electron_vidNestedWPBitmap")
            mats = re.search(r'\((.*?)\)',eledoc).groups()
            eleCuts = mats[0].split(",")
            nbit = int(re.findall(r'(\d+) bits',eledoc)[0])
            testbit = (1<<nbit)-1
            elebits = DataFrame()
            skipflag = False
            # newID = { k:False for k ,v in  dict_eleID.items() }
            for i, c in enumerate(eleCuts):
                if c == 'GsfEleRelPFIsoScaledCut':
                    skipflag=True
                    continue
                ibit = elearr  >> (nbit * i) & testbit
                elebits[i+1-skipflag] = ibit
            bitframe, tempframe = DataFrame(), DataFrame()
            for i in range(elecs.pt.shape[1]):
                for c in elebits.columns:
                    tempframe[c] = elebits[c].str[i]
                bitframe[i+1] = tempframe.min(axis=1)
            elecs.bitframe = bitframe


            elecs.trimto(sigjets.pt)
            Event(elecs).sync()

            belecs = []
            for i in range(len(bgjets)):
                belecs.append(PhysObj('Electron',bgfiles[i],'pt','eta','phi','miniPFRelIso_all','sip3d','mvaFall17V1Iso_WP80','mvaTTH'))
                eframe = fastdR(bgjets[i], belecs[i])
                belecs[i].jetdr = eframe[belecs[i].pt != 0].dropna(how='all')

                eledoc = bgevents[i].get("Electron_vidNestedWPBitmap").title.decode()
                elearr = bgevents[i].array("Electron_vidNestedWPBitmap")
                mats = re.search(r'\((.*?)\)',eledoc).groups()
                eleCuts = mats[0].split(",")
                nbit = int(re.findall(r'(\d+) bits',eledoc)[0])
                testbit = (1<<nbit)-1
                elebits = DataFrame()
                skipflag = False
                # newID = { k:False for k ,v in  dict_eleID.items() }
                for j, c in enumerate(eleCuts):
                    if c == 'GsfEleRelPFIsoScaledCut':
                        skipflag=True
                        continue
                    ibit = elearr  >> (nbit * j) & testbit
                    elebits[j+1-skipflag] = ibit
                bitframe, tempframe = DataFrame(), DataFrame()
                for j in range(belecs[i].pt.shape[1]):
                    for c in elebits.columns:
                        tempframe[c] = elebits[c].str[j]
                    bitframe[j+1] = tempframe.min(axis=1)
                belecs[i].bitframe = bitframe

                belecs[i].trimto(bgjets[i].pt)
                Event(belecs[i]).sync()
                
            # import pdb; pdb.set_trace()

            for elec in belecs + [elecs]:
                elec.cut(elec.bitframe == 4)
                elec.cut(elec.miniPFRelIso_all < 0.1)
                elec.cut(elec.pt > 30)
                elec.cut(abs(elec.eta) < 2.5)
                elec.cut(abs(elec.sip3d) < 4.0)
                elec.cut(elec.jetdr > 0.8)
                ## HEM special electron veto
                elec.cut(~((elec.eta < -1.4) & (elec.phi < -0.87) & (elec.phi > -1.57)))
                elec.cut(elec.mvaTTH > -0.4)
                ## Only keep the highest passing pT electron
                elec.cut(elec.pt.rank(axis=1,method='first',ascending=False) == 1)
                ## Electron crunch
                for prop in elec:
                    elec[prop] = DataFrame(elec[prop].max(axis=1)).rename(columns=inc)
            
    

            # import pdb; pdb.set_trace()

            sigevv.nlep = DataFrame(np.isfinite(muons.pt).sum(axis=1).add(np.isfinite(elecs.pt).sum(axis=1),fill_value=0)).rename(columns=inc)
            sigevv.cut(sigevv.nlep > 0)
            for i in range(len(bgjets)):
                bgevv[i].nlep = DataFrame(np.isfinite(bmuons[i].pt).sum(axis=1).add(np.isfinite(belecs[i].pt).sum(axis=1),fill_value=0)).rename(columns=inc)
                bgevv[i].cut(bgevv[i].nlep > 0)
        
            for e in [ev] + bev:
                e.sync()
            
            ## TODO: potentially expand to cover signal but not data
            musfid = uproot.open('Muon_IDScaleFactor_wSys_2018.root')['NUM_MediumID_DEN_TrackerMuons_pt_abseta']
            musfiso= uproot.open('Muon_MediumID_MiniIso0p2SF_2017.root')['TnP_MC_NUM_MiniIso02Cut_DEN_MediumID_PAR_pt_eta']
            musfidx = musfid.edges[0]
            musfidy = musfid.edges[1]
            musfisox = musfiso.edges[0]
            musfisoy = musfiso.edges[1]
            for muon in bmuons:
                muon.extweight = muon.pt / muon.pt
                for xi in range(len(musfidx)-1):
                    for yi in range(len(musfidy)-1):
                        muon.extweight[(muon.pt >= musfidx[xi]) & (muon.pt < musfidx[xi+1]) &
                                        (muon.eta>= musfidy[yi]) & (muon.eta< musfidy[yi-1])] *= musfid.values[xi][yi] 
                for xi in range(len(musfisox)-1):
                    for yi in range(len(musfisoy)-1):
                        muon.extweight[(muon.pt >= musfisox[xi]) & (muon.pt < musfisox[xi+1]) &
                                        (muon.eta>= musfisoy[yi]) & (muon.eta< musfisoy[yi-1])] *= musfiso.values[xi][yi]
                        
            elsfcut = uproot.open('Electron_SUSYScaleFactors_2017v2ID_Run2018.root')['Run2018_CutBasedTightNoIso94XV2']
            elsfiso = uproot.open('Electron_SUSYScaleFactors_2017v2ID_Run2018.root')['Run2018_Mini']
            elsfgam = uproot.open('Electron_GT10GeV_RecoSF_2017v2ID_Run2018.root')['EGamma_SF2D']
    
    
            elsfgamx = elsfgam.edges[0]
            elsfgamy = elsfgam.edges[1]
            elsfisox = elsfiso.edges[0]
            elsfisoy = elsfiso.edges[1]
            elsfcutx = elsfcut.edges[0]
            elsfcuty = elsfcut.edges[1]
    
            for elec in belecs:
                elec.extweight = elec.pt / elec.pt
                for xi in range(len(elsfgamx)-1):
                    for yi in range(len(elsfgamy)-1):
                        elec.extweight[(elec.pt >= elsfgamy[yi]) & (elec.pt < elsfgamy[yi+1]) &
                                        (elec.eta>= elsfgamx[xi]) & (elec.eta< elsfgamx[xi+1])] *= elsfgam.values[xi][yi]
                for xi in range(len(elsfisox)-1):
                    for yi in range(len(elsfisoy)-1):
                        elec.extweight[(elec.pt >= elsfisoy[yi]) & (elec.pt < elsfisoy[yi+1]) &
                                        (elec.eta>= elsfisox[xi]) & (elec.eta< elsfisox[xi+1])] *= elsfiso.values[xi][yi]
                for xi in range(len(elsfcutx)-1):
                    for yi in range(len(elsfcuty)-1):
                        elec.extweight[(elec.pt >= elsfcuty[yi]) & (elec.pt < elsfcuty[yi+1]) &
                                        (elec.eta>= elsfcutx[xi]) & (elec.eta< elsfcutx[xi+1])] *= elsfcut.values[xi][yi]
            
            for i in range(len(bgevv)):
                if bgevv[i].extweight.shape[0]:
                    bgevv[i].extweight[1][bgevv[i].extweight[1]*bmuons[i].extweight.max(axis=1) > 0] *= bmuons[i].extweight.max(axis=1)
                    bgevv[i].extweight[1][bgevv[i].extweight[1]*belecs[i].extweight.max(axis=1) > 0] *= belecs[i].extweight.max(axis=1)
                    belecs[i].eta = abs(belecs[i].eta)
            elecs.eta = abs(elecs.eta)
            
            ev.register(sigsj)
            ev.sync()
            for i in range(len(bev)):
                bev[i].register(bgsj[i])
                bev[i].sync()
                bgsj[i].fatdr = fastdR(bgjets[i],bgsj[i])         
            sigsj.fatdr = fastdR(sigjets,sigsj)
            
            for sj in [sigsj] + bgsj:
                sj.cut(sj.pt > 30)
                sj.cut(abs(sj.eta) < 2.4)
                sj.cut(sj.puId >= 1)
                ## HEM veto for jets
                sj.cut((sj.eta > -1.37) | (sj.phi < -1.77) | (sj.phi > -0.67))
                sj.cut(sj.fatdr > 0.8)
            
            for i in range(len(bev)):
                bmuons[i].trimTo(bgsj[i].pt)
                belecs[i].trimTo(bgsj[i].pt)
            muons.trimTo(sigsj.pt)
            elecs.trimTo(sigsj.pt)
            
            # import pdb; pdb.set_trace()
            ## fill invalid dR values to remove veto power
            mudr  = ((sigsj.pt*0) + fastdR(muons,sigsj)).fillna(99)
            eldr  = ((sigsj.pt*0) + fastdR(elecs,sigsj)).fillna(99)
            ## Take the lowest dR of any position (or 0 if none valid), and check if it's below the cut.
            sigsj.cut(  (mudr[mudr < eldr].fillna(0) + eldr[eldr <= mudr].fillna(0))  > 0.4)
            print("Signal dR\n----------------------------------")
            print((mudr[mudr < eldr].fillna(0) + eldr[eldr <= mudr].fillna(0)))
            for i in range(len(bgsj)):
                bmudr = ((bgsj[i].pt*0) + fastdR(bmuons[i],bgsj[i])).fillna(99)
                beldr = ((bgsj[i].pt*0) + fastdR(belecs[i],bgsj[i])).fillna(99)
                ## Take the lowest dR of any position (or 0 if none valid), and check if it's below the cut.
                bgsj[i].cut(  (bmudr[bmudr < beldr].fillna(0) + beldr[beldr <= bmudr].fillna(0))  > 0.4)
                
            for sj in [sigsj] + bgsj:
                ## Delete the highest jet, to leave only events with 2+ jets remaining
                sj.cut(sj.pt.rank(axis=1,method='first',ascending=False) > 1)
            print("Signal AK4 pT\n----------------------------------")
            print(sigsj.pt)
            for e in bev + [ev]:
                e.sync()
            

        ##################################
        # Preparing Neural Net Variables #
        ##################################
        # if LOADMODEL and (not passplots):
            # bgjetframe['extweight'] = lumipucalc(bgjetframe)
            # bgrawframe['extweight'] = lumipucalc(bgrawframe)

        # import pdb; pdb.set_trace()


        for i in range(len(bgevv)):
            if 'genweight' in bgevv[i].keys():
                bgevv[i].extweight[bgevv[i].genweight < 0] *= -1
        if dataflag != 1:
            sigevv.extweight[sigevv.genweight < 0] *= -1

        bgjetframe = DataFrame()
        extvars = ['event','extweight','npvs','npvsG']
        if 'pt' in netvars: ptfix = ['phi']
        else: ptfix = ['phi','pt']
        if LOADMODEL and (not passplots) and ('lepton' in REGION):
            muvars = ['pt','eta','miniPFRelIso_all','sip3d','jetdr']
            elvars = ['e' + x for x in muvars]
            muvars = ['m' + x for x in muvars]
            extvars += ['metpt', 'nlep']
            for obj in [muons] + [elecs]:
            # for obj in [elecs]:
                obj.trimto(sigjets.pt)
            for i in range(len(bgjets)):
                bmuons[i].trimto(bgjets[i].pt)
                belecs[i].trimto(bgjets[i].pt)
        else:
            muvars = []
            elvars = muvars
            extvars += ['metpt']
        if isLHE:
            bgpieces = []
            wtpieces = []

            for i in range(nlhe):
                tempframe = DataFrame()
                twgtframe = DataFrame()
                for prop in netvars + ptfix:
                    twgtframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                for prop in extvars:
                    twgtframe[prop] = bgevv[i][prop].max(axis=1)
                for prop in muvars:
                    twgtframe[prop] = bmuons[i][prop[1:]][bmuons[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                for prop in elvars:
                    twgtframe[prop] = belecs[i][prop[1:]][belecs[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                if 'eta' in netvars:
                    twgtframe['eta'] = abs(twgtframe['eta'])

                if np.max(lheweights) > 1.0:
                    tempframe = twgtframe.sample(frac=lheweights[i],random_state=6,replace=True)
                else:
                    tempframe = twgtframe.sample(frac=lheweights[i],random_state=6)
                bgpieces.append(tempframe)
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(bgpieces,ignore_index=True)
            bgrawframe = pd.concat(wtpieces,ignore_index=True)
            # bgjetframe = bgjetframe.dropna()
            # bgrawframe = bgrawframe.dropna()
            bgjetframe['val'] = 0
            bgrawframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]

        else:
            for prop in netvars + ptfix:
                bgjetframe[prop] = bgjets[0][prop][bgjets[0]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            for prop in extvars:
                bgjetframe[prop] = bgevv[0][prop][1]
            for prop in muvars:
                bgjetframe[prop] = bmuons[0][prop[1:]][bmuons[0]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            for prop in elvars:
                bgjetframe[prop] = belecs[0][prop[1:]][belecs[0]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)

            if 'eta' in netvars:
                bgjetframe['eta'] = abs(bgjetframe['eta'])
            bgrawframe = bgjetframe

            bgjetframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]

        #nbg = bgtrnframe.shape[0]

        sigjetframe = DataFrame()
        for prop in netvars + ptfix:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        for prop in extvars:
            sigjetframe[prop] = sigevv[prop]
        for prop in muvars:
            sigjetframe[prop] = muons[prop[1:]][muons['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        for prop in elvars:
            sigjetframe[prop] = elecs[prop[1:]][elecs['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)

        if 'eta' in netvars:
            sigjetframe['eta'] = abs(sigjetframe['eta'])

        ## Signal MC specific pt reweighting
        # if dataflag != 1:
        #     ptwgt = 3.9 - (0.4*np.log2(sigjetframe.pt))
        #     ptwgt[ptwgt < 0.1] = 0.1
        #     sigjetframe.extweight = sigjetframe.extweight * ptwgt

        sigjetframe['val'] = 1
        sigtrnframe = sigjetframe[sigjetframe['event']%2 == 0]

        # import pdb;
        # pdb.set_trace()

        print(f"{Skey} cut to {sigjetframe.shape[0]} entries")
        print(f"{Bkey} has {bgrawframe.shape[0]} entries")



        # bgjetframe = bgjetframe.fillna(0)
        # bgrawframe = bgrawframe.fillna(0)
        # sigjetframe = sigjetframe.fillna(0)

        # bgjetframe = bgjetframe.reset_index(drop=True)
        # sigjetframe = sigjetframe.reset_index(drop=True)

        # extvars = extvars + muvars + elvars + ['val'] + ['phi']

        #######################
        # Training Neural Net #
        #######################


        if LOADMODEL and not TUTOR:
            if isLHE:
                bgjetframe=bgrawframe
            ## Normalize event number between QCD and data samples
            # if dataflag == True:
            #     bgjetframe['extweight'] = bgjetframe['extweight'] * np.sum(sigjetframe['extweight'])/np.sum(bgjetframe['extweight'])

            X_inputs = pd.concat([bgjetframe,sigjetframe],ignore_index=True)
            W_inputs = X_inputs['extweight']
            Y_inputs = X_inputs['val']
            X_inputs = X_inputs.drop(set(bgjetframe.columns) - set(netvars),axis=1)
            model = keras.models.load_model(f"archive/{mname}.hdf5", compile=False) #archive
            scaler = pickle.load( open(f"archive/{sname}.p", "rb" ) )
            # model = keras.models.load_model('archive/C.hdf5', compile=False) #archive
            # scaler = pickle.load( open("archive/C.p", "rb" ) )

            X_inputs = scaler.transform(X_inputs)
            if disconame and not (subnet or NET=="B"):
                pickle.dump(model.predict(X_inputs),open(f"RUN-{disconame}.p",'wb'))

        else:
            X_test = pd.concat([bgjetframe.drop(bgtrnframe.index),sigjetframe.drop(sigtrnframe.index)])
             ##
            if bgtrnframe.shape[0] < sigtrnframe.shape[0]:
                bgtrnlst = []
                for i in range(int(np.floor(sigtrnframe.shape[0] / bgtrnframe.shape[0]))):
                    bgtrnlst.append(bgtrnframe)
                bgtrnframe = pd.concat(bgtrnlst,ignore_index=True)
                print (f"bg now {bgtrnframe.shape[0]} to sg {sigtrnframe.shape[0]}")

            X_train = pd.concat([bgtrnframe,sigtrnframe],ignore_index=True)
            W_test = X_test['extweight']
            W_train = X_train['extweight']
            Y_test = X_test['val']
            Y_train= X_train['val']
            X_test = X_test.drop(set(bgjetframe.columns) - set(netvars),axis=1)
            X_train = X_train.drop(set(bgjetframe.columns) - set(netvars),axis=1)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if TUTOR == True:
                tutor(X_train,X_test,Y_train,Y_test)
                sys.exit()
            elif subnet or NET=="B":
                discotr = pickle.load(open(f"archive/TR-{disconame}.p",'rb'))
                discote = pickle.load(open(f"archive/TE-{disconame}.p",'rb'))
                X_train = [X_train, discotr]
                X_test = [X_test, discote]
                history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True,verbose=False)
                rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test)[:,0].ravel())
                trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train)[:,0].ravel())
                test_loss, test_acc = model.evaluate(X_test, Y_test)
                print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
                model.save(f"{mname}.hdf5")
                pickle.dump(scaler, open(f"{sname}.p", "wb"))
                plt.clf()
                plt.plot([0,1],[0,1],'k--')
                plt.plot(rocx,rocy,'red')
                plt.plot(trocx,trocy,'b:')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(['y=x','Validation','Training'])
                plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
                plt.savefig(plots['Distribution'].fname+'ROC')
                sys.exit()

            else:
                history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True,verbose=False)
                rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
                trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train).ravel())
                test_loss, test_acc = model.evaluate(X_test, Y_test)
                print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
                if disconame:
                    pickle.dump(model.predict(X_train),open(f"TR-{disconame}.p",'wb'))
                    pickle.dump(model.predict(X_test),open(f"TE-{disconame}.p",'wb'))
                    model.save(f"{mname}.hdf5")
        if REGION: passnum = 1
        else: passnum = 0.6
        ##################################
        # Analyzing and Plotting Outputs #
        ##################################
        
        ## Skip everything if we're out of events
        if not sigjetframe.shape[0]:
            print("Out of signal events, stopping to prevent nonsensical output")
            return 0
        if not bgjetframe.shape[0]:
            print("Out of background events, skipping to next file")
            continue
        

        varlist = netvars + muvars + elvars + ['npvs','npvsG','metpt']#,'nlep']
        
        if LOADMODEL:
            if subnet or NET=="B":
                diststt = model.predict([X_inputs[Y_inputs==1],Y_inputs[Y_inputs==1]])[:,0]
                distbtt = model.predict([X_inputs[Y_inputs==0],Y_inputs[Y_inputs==0]])[:,0]
            else:
                diststt = model.predict(X_inputs[Y_inputs==1])
                distbtt = model.predict(X_inputs[Y_inputs==0])
        else:
            diststr = model.predict(X_train[Y_train==1])
            distste = model.predict(X_test[Y_test==1])
            distbtr = model.predict(X_train[Y_train==0])
            distbte = model.predict(X_test[Y_test==0])
            diststt = model.predict(scaler.transform(sigjetframe.drop(set(sigjetframe.columns) - set(netvars),axis=1)))
            distbtt = model.predict(scaler.transform(bgjetframe.drop(set(bgjetframe.columns) - set(netvars),axis=1)))


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

            # #############

            ## Store the output confidence values and their weights
            distbf = DataFrame(distbtt)
            distbf['W'] = W_inputs[Y_inputs==0].reset_index(drop=True)
            ## Sort both of them together by the confidence values
            distbf = distbf.sort_values(by=[0])
            ## Store the cumulative sum of the weights
            distbf['W_csum'] = distbf.W.cumsum()
            ## Decide what weighted interval each volume should encompass
            isize = distbf.W_csum.max() / 10
            ## label which of those intervals each event falls into
            distbf['bin'] = (distbf.W_csum / isize).dropna().apply(math.floor)
            temp = []
            for i in range(1,10):
                temp.append(distbf[distbf['bin']==i-1][0].max())

            saveball = {
                'distbf':distbf,
                'distsf':distsf,
                "SensS":plots["SensS"][1],
                "SensB":temp,
                "WS":W_inputs[Y_inputs==1],
                "WB":W_inputs[Y_inputs==0],
                }
            

            # for i in range(1,10):
            #     plots['SensS'][1][i] = distbf[distbf['bin']==i-1][0].max()
            #     plots['SensB'][1][i] = plots['SensS'][1][i]


            # plots['SensS'].fill(diststt,W_inputs[Y_inputs==1])
            # plots["SensB"].fill(distbtt,W_inputs[Y_inputs==0])

            # ########

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


            plt.clf()
            plt.plot([0,1],[0,1],'k--')
            plt.plot(rocx,rocy,'red')
            plt.plot(trocx,trocy,'b:')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(['y=x','Validation','Training'])
            plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
            plt.savefig(plots['Distribution'].fname+'ROC')


        for col in varlist:
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


    for p in [plots['DistStr'],plots['DistSte'],plots['DistBtr'],plots['DistBte'],plots['SensB'],plots['SensS']]:
        if sum(p[0]) != 0:
            p.ival = sum(p[0])
            p.ndivide(sum(p[0]))

    ## Blinding control
    if dataflag == True:
        for i in range(len(plots['DistSte'][0])):
            if plots['DistSte'][1][i] >= passnum:
                plots['DistSte'][0][i] = 0
                plots['DistSte'].ser[i] = 0

    if LOADMODEL:
        leg = [Sname,Bname]
        if 'Combined' in Bname:
            leg = [Sname,'Combined MC']
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
        plt.clf()
        plots['SensS'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['SensB'].xlabel = f"Confidence (S={Sens})"
        plots['SensB'].plot(legend=leg,same=True,linestyle=':',logv=True,ylim=(0.00001,1),**plargs[Bkey])
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

        for col in netvars + muvars + ['npvs','npvsG']:
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
        for col in varlist:
            plt.clf()
            vplots['SG'+col].make(linestyle='-',**plargs[Skey])
            vplots['BG'+col].make(linestyle=':',**plargs[Bkey],error=dataflag)
            vplots[col].plot(same=True,legend=[f"{Bname} ({round(vplots['BG'+col][0].sum())})",f"{Sname} ({round(vplots['SG'+col][0].sum())})"])

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

    cplot.plot()
    pickle.dump(cplot, open('cutplot.p','wb'))


    if not LOADMODEL:
        model.save('weighted.hdf5')
        pickle.dump(scaler, open(f"{sname}.p", "wb"))
        
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
        folder = plots['Distribution'].fname.split('/')[0]+'/'
        if setname:
            if subf:
                setname = f"{subf}/{setname}"
            pickle.dump(arcdict, open(f"{folder}{setname} {NET}.p", "wb"))
            pickle.dump(diststt, open(f"{folder}{setname} diststt{NET}.p", 'wb'))
            pickle.dump(distbtt, open(f"{folder}{setname} distbtt{NET}.p", 'wb'))
            pickle.dump(saveball,open(f"{folder}{setname} saveball{NET}.p",'wb'))
        else: pickle.dump(arcdict, open(f"{folder}arcdict{NET}.p", "wb"))
        # pickle.dump(sigjetframe,open('sigjetframe.p','wb'))
        # pickle.dump(bgjetframe,open('bgjetframe.p','wb'))
        


        # datalist = []
        # for frame in [sigjetframe.drop(set(sigjetframe.columns) - set(netvars),axis=1),
        #               bgjetframe.drop(set(bgjetframe.columns) - set(netvars),axis=1)]:
        #     datalist.append(spearmanr(frame).correlation)
        # datalist.append(datalist[0] - datalist[1])

        # mindata = []
        # fnames = ['otherplots/SGSpearmanD','otherplots/BGSpearmanD']
        # for k in range(2):
        #     distdata = datalist[k] * 0
        #     for x in range(distdata.shape[0]):
        #         for y in range(distdata.shape[1]):
        #             distdata[y,x] = min((np.sum((datalist[k][y] - datalist[k][x])**2))**.5,
        #                                 (np.sum((datalist[k][y] + datalist[k][x])**2))**.5)
        #     mindata.append(distdata)

        #     plt.clf()
        #     plt.rcParams.update({'font.size': 14})
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        #     linkage = hierarchy.ward(squareform(mindata[k]))
        #     dendro = hierarchy.dendrogram(
        #         linkage, labels=netvars, ax=ax1, leaf_rotation=90
        #     )
        #     dendro_idx = np.arange(0, len(dendro['ivl']))
        #     imdata = mindata[k][dendro['leaves'], :][:, dendro['leaves']]
        #     ax2.imshow(imdata)
        #     if True:
        #         strarray = imdata.round(3).astype(str)
        #         for i in range(len(imdata[0])):
        #             for j in range(len(imdata[1])):
        #                 plt.text(i,j, strarray[i,j],color="w", ha="center", va="center", fontweight='normal',fontsize=11).set_path_effects([PathEffects.withStroke(linewidth=2,foreground='k')])
        #     ax2.set_xticks(dendro_idx)
        #     ax2.set_yticks(dendro_idx)
        #     ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        #     ax2.set_yticklabels(dendro['ivl'])
        #     fig.tight_layout()
        #     #plt.show()
        #     plt.gcf()
        #     plt.savefig(fnames[k])

        # fnames = ['otherplots/SGSpearman','otherplots/BGSpearman','otherplots/diff2D']
        # for k in range(3):
        #     plt.clf()
        #     plt.rcParams.update({'font.size': 14})
        #     fig, ax1 = plt.subplots(1, 1, figsize=(16, 16))
        #     imdata = datalist[k][dendro['leaves'], :][:, dendro['leaves']]
        #     ax1.imshow(imdata)
        #     if True:
        #         strarray = imdata.round(3).astype(str)
        #         for i in range(len(imdata[0])):
        #             for j in range(len(imdata[1])):
        #                 plt.text(i,j, strarray[i,j],color="w", ha="center", va="center", fontweight='normal',fontsize=11).set_path_effects([PathEffects.withStroke(linewidth=2,foreground='k')])
        #     ax1.set_xticks(dendro_idx)
        #     ax1.set_yticks(dendro_idx)
        #     ax1.set_xticklabels(dendro['ivl'], rotation='vertical')
        #     ax1.set_yticklabels(dendro['ivl'])
        #     fig.tight_layout()
        #     #plt.show()
        #     plt.gcf()
        #     plt.savefig(fnames[k])



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
        NET = "F"
        subf = ''
        ## Check for file sources
        for i in range(nrgs):
            arg = sys.argv[i]
            if '-s' in arg:
                sigfile = sys.argv[i+1]
            elif '-b' in arg:
                bgfile = sys.argv[i+1]
            elif ('-n' in arg) or ('-N' in arg):
                NET = sys.argv[i+1]
            elif ('-Pass' in arg) or ('-pass' in arg):
                passplots = True
            elif ('-Train' in arg) or ('-train' in arg):
                LOADMODEL=False
            elif ('-Tutor' in arg) or ('-tutor' in arg):
                TUTOR = True
                LOADMODEL=False
            elif ('-F' in arg) or ('-f' in arg):
                subf = sys.argv[i+1]
            #else: dialogue()
        #print('-')
        #print('sigfiles',sigfiles,'datafiles',datafiles)
        #print('-')
        if not sigfile or not bgfile:
            dialogue()
        else:
            ana(sigfile,bgfile,LOADMODEL,TUTOR,passplots,NET,subf)

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