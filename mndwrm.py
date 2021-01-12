#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
### Compiled with Keras-2.3.1 Tensorflow-1.14.0                      ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

from ROOT import TH1F, TFile, gROOT, TCanvas

import sys, math
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist, PhysObj, Event, inc, fstrip, InputConfig#, Hist2D
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

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)


import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()

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
    
## Takes a dataframe containing the 'extweight' 
def lumipucalc(inframe):
    for var in ['extweight','mpt','meta','mip','npvsG']:
        if var not in inframe.columns:
            raise ValueError(f"Dataframe passed to lumipucalc() with no {var} column")
    Rtensor = pickle.load(open('MuonRtensor.p',"rb"))
    Ltensor = pickle.load(open('MuonLtensor.p',"rb"))
    Rmeta = Rtensor.pop('meta')
    Lmeta = Ltensor.pop('meta')
    for list in Rmeta['x'],Rmeta['y'],Lmeta['x'],Lmeta['y']:
        list[-1] = 999999
#    if Rmeta['x'] != Lmeta['x']:
#        raise ValueError("mismatched tensor axis detected")
        
    for xi in range(len(Rmeta['x'])-1):
        for yi in range(len(Rmeta['y'])-1):
            x  = Rmeta['x'][xi]
            xn = Rmeta['x'][xi+1]
            y  = Rmeta['y'][yi]
            yn = Rmeta['y'][yi+1]
            for b in range(len(Rtensor[x][y]['L'])):
                inframe['extweight'][(inframe['mpt'] > x)&(inframe['mpt'] < xn)&\
                       (inframe['mip'] > y)&(inframe['mip'] < yn)&(inframe['meta'] < 1.5)&\
                       (inframe['npvsG'] == b+1)] = inframe['extweight'] * Rtensor[x][y]['L'][b] * Ltensor[x][y]['L']
            
                inframe['extweight'][(inframe['mpt'] > x)&(inframe['mpt'] < xn)&\
                (inframe['mip'] > y)&(inframe['mip'] < yn)&(inframe['meta'] >= 1.5)&\
                (inframe['npvsG'] == b+1)] = inframe['extweight'] * Rtensor[x][y]['H'][b] * Ltensor[x][y]['H']
    return inframe['extweight']
  
def computedR(jet,thing,nms=['jet','thing']):
    nj = jet.eta.shape[1]
    nt = thing.eta.shape[1]
    ## Create our dR dataframe by populating its first column and naming it accordingly
    jtdr2 = pd.DataFrame(np.power(jet.eta[1] - thing.eta[1],2) + np.power(jet.phi[1] - thing.phi[1],2)).rename(columns={1:f"{nms[0]} 1 {nms[1]} 1"})
    jtstr = []
    ## Loop over jet x thing combinations
    for j in range(1,nj+1):
        for t in range(1,nt+1):
            jtstr.append(f"{nms[0]} {j} {nms[1]} {t}")
            if (j+t==2):
                continue
            jtdr2[jtstr[-1]] = pd.DataFrame(np.power(jet.eta[j]-thing.eta[t],2) + np.power(jet.phi[j]-thing.phi[t],2))
    return np.sqrt(jtdr2)
        
#%%

def ana(sigfiles,bgfiles,LOADMODEL=True,TUTOR=False,passplots=False):
    #%%
    passplots=False
    ic = InputConfig('GGH_HPT.json','bgC.json')
    if ic.sigdata:
        dataflag = -1
        if ic.bgdata:
            raise ValueError("Both signal and background are marked as data")
    elif ic.bgdata:
        dataflag = 1
    else: dataflag = 0
    sigfiles = ic.sigfiles
    bgfiles = ic.bgfiles
    if ic.bglhe:
        isLHE=True
        lheweights = ic.bgweight
        nlhe = len(lheweights)
    else: isLHE=False
    ###################
    # Plots and Setup #
    ###################
    #training=True
    #training=False
    #tf.random.set_random_seed(2)
    #tf.compat.v1.set_random_seed(2)
    #np.random.seed(2)
     
    #fig = plt.figure(figsize=(10.0,6.0))
    
    # if ic.signame:
    #     namedict.update({"Signal":ic.signame})
    # elif ic.sigdata:
    #     namedict.update({"Signal":"Data"})
    # else: namedict.update({"Signal":"Signal"})
    # if ic.bgname:
    #     namedict.update({"Background":ic.bgname})
    # elif ic.bgdata:
    #     namedict.update({"Background":"Data"})
    # else: namedict.update({"Background":"Background"})
    # for i in range(3):
    #     if names[i]:
    #         namedict.update({namelist[i]:names[i]})
    #     else:
    #         namedict.update({namelist[i]:namelist[i]})
    
    #global LOADMODEL
    plargs = {'Data':  {'color':'black','htype':'step'},
              'Background':   {'color':'red','htype':'step'},
              'Signal':{'color':'blue','htype':'step'}
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
            # Skey = 'Data'
            # if names[2]:
            #     Sname = names[2]
            # else: Sname = Skey
        # else:
        #     Bkey = 'Data'
        #     if names[2]:
        #         Bname = names[2]
        #     else: Bname = Bkey
    if TUTOR:
        LOADMODEL = False
        
    if not LOADMODEL:
        passplots = False

    netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','n2b1','submass1','submass2','subtau1','subtau2','nsv']
    #netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs']
    #netvars = ['DeepB','H4qvs','DDBvL','CSVV2']
    #netvars = ['pt','eta','mass','msoft']

    
    l1 = 8
    l2 = 8
    l3 = 8
    alpha = 0.5
    gamma = 2.2
    model = keras.Sequential([
            #keras.Input(shape=(4,),dtype='float32'),
            #keras.layers.Flatten(input_shape=(8,)),
            keras.layers.Dense(l1, activation=tf.nn.relu,input_shape=(len(netvars),)),
            keras.layers.Dense(l2, activation=tf.nn.relu),
            keras.layers.Dense(l3, activation=tf.nn.relu),
            #keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ])
    optimizer  = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,#'adam',     
                  #loss='binary_crossentropy',
                  #loss=[focal_loss],
                  #loss=[custom],
                  loss=[binary_focal_loss(alpha, gamma)],
                  metrics=['accuracy'])#,tf.keras.metrics.AUC()])
            
    #nbatch = math.floor(nbg / (2*nsig))
    
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
        "mass":     Hist(50 ,(0,200)    ,'mass for highest pT jet','Fractional Distribution','netplots/pmass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet in all (red), passing signal (blue), and signal (black) events','Fractional Distribution','netplots/pCSVV2'),
        "DeepB":    Hist(22 ,(0,1.1)    ,'DeepB for highest pT jet','Fractional Distribution','netplots/pDeepB'),
        "msoft":    Hist(50 ,(0,200)    ,'msoft for highest pT jet in all (red), passing (blue), and failing  (black) events','Fractional Distribution','netplots/pmsoft'),
        "DDBvL":    Hist(55 ,(0,1.1)    ,'DDBvL for highest pT jet','Fractional Distribution','netplots/pDDBvL'),
        "H4qvs":    Hist(24 ,(-10,2)    ,'H4qvs for highest pT jet','Fractional Distribution','netplots/pH4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/pnpvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/pnpvsG'),
        "mpt":      Hist(80 ,(150,550)  ,'pT for highest pT muon','Fractional Distribution','netplots/pmpt'),
        "meta":     Hist(15 ,(0,3)      ,'|eta| for highest pT muon','Fractional Distribution','netplots/pmeta'),
        "mip":      Hist(20 ,(2,12)     ,'dxy/dxyError for highest pT muon','Fractional Distribution','netplots/pmip'),
        "HT":       Hist(500 ,(0,5000)  ,'pT for highest pT jet','Fractional Distribution','netplots/pHT'),
        "n2b1":     Hist(10 ,(0,1)     ,'n2b1 for highest pT jet','Fractional Distribution','netplots/pn2b1'),
        "submass1": Hist(105,(-5,105)   ,'submass for 1st subjet of highest pT jet','Fractional Distribution','netplots/psubmass1'),
        "submass2": Hist(105,(-5,105)   ,'submass for 2nd subjet of highest pT jet','Fractional Distribution','netplots/psubmass2'),
        "subtau1":  Hist(10 ,(0,1)     ,'subtau1 for 1st subjet of highest pT jet','Fractional Distribution','netplots/psubtau1'),
        "subtau2":  Hist(10 ,(0,1)     ,'subtau1 for 2nd subjet of highest pT jet','Fractional Distribution','netplots/psubtau2'),
        'nsv':      Hist(20 ,(0,20)     ,'# of secondary vertices with dR<0.8 to highest pT jet','Fractional Distribution','netplots/pnsv'),
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
        "pt":       Hist(80 ,(150,550)  ,'pT for highest pT jet','Fractional Distribution','netplots/pt'),
        "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT jet','Fractional Distribution','netplots/eta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/phi'),
        "mass":     Hist(50 ,(0,200)    ,'mass for highest pT jet','Fractional Distribution','netplots/mass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet','Fractional Distribution','netplots/CSVV2'),
        "DeepB":    Hist(22 ,(0,1.1)    ,'DeepB for highest pT jet','Fractional Distribution','netplots/DeepB'),
        "msoft":    Hist(50 ,(0,200)    ,'msoft for highest pT jet','Fractional Distribution','netplots/msoft'),
        "DDBvL":    Hist(55 ,(0,1.1)    ,'DDBvL for highest pT jet','Fractional Distribution','netplots/DDBvL'),
        "H4qvs":    Hist(20 ,(0,1)      ,'H4qvs for highest pT jet','Fractional Distribution','netplots/H4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/npvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/npvsG'),
        "mpt":      Hist(80 ,(150,550)  ,'pT for highest pT muon','Fractional Distribution','netplots/mpt'),
        "meta":     Hist(15 ,(0,3)      ,'|eta| for highest pT muon','Fractional Distribution','netplots/meta'),
        "mip":      Hist(20 ,(2,12)     ,'dxy/dxyError for highest pT muon','Fractional Distribution','netplots/mip'),
        "HT":       Hist(500,(0,5000)   ,'pT for highest pT jet','Fractional Distribution','netplots/HT'),
        "n2b1":     Hist(10 ,(0,1)     ,'n2b1 for highest pT jet','Fractional Distribution','netplots/n2b1'),
        "submass1": Hist(105,(-5,105)   ,'submass for 1st subjet of highest pT jet','Fractional Distribution','netplots/submass1'),
        "submass2": Hist(105,(-5,105)   ,'submass for 2nd subjet of highest pT jet','Fractional Distribution','netplots/submass2'),
        "subtau1":  Hist(10 ,(0,1)     ,'subtau1 for 1st subjet of highest pT jet','Fractional Distribution','netplots/subtau1'),
        "subtau2":  Hist(10 ,(0,1)     ,'subtau1 for 2nd subjet of highest pT jet','Fractional Distribution','netplots/subtau2'),
        'nsv':      Hist(20 ,(0,20)     ,'# of secondary vertices with dR<0.8 to highest pT jet','Fractional Distribution','netplots/nsv'),
        
    }
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
    # if isLHE:
    #     nbg = len(bgfiles)/nlhe
    #     if float(nbg).is_integer():
    #         nbg = int(nbg)
    #     else:
    #         raise Exception('LHE argument specified, but BG files do not divide evenly into '+str(nlhe))
    # else:
    #     nbg = len(bgfiles)
    # nsig = len(sigfiles)
    # sigmbg = nbg - nsig
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
        
        # ## Loop some data if the bg/signal files need to be equalized
        # if sigmbg > 0:
        #     print('Catching up signal')
        #     sigfiles.append(sigfiles[fnum])
        #     sigmbg = sigmbg - 1
        # elif sigmbg < 0:
        #     if isLHE:
        #         print('Catching up background')
        #         for i in range(nlhe):
        #             bgfiles.append(bgfiles[fnum+i])
        #             sigmbg = sigmbg + 1
        #     else:
        #         print('Catching up background')
        #         bgfiles.append(bgfiles[fnum])
        #         sigmbg = sigmbg + 1
        # print('diff:',sigmbg)
        ## Open our file and grab the events tree
        if isLHE:
            bgevents = []
            for i in range(nlhe):
                idx = fnum*nlhe + i
                print('Opening ',bgfiles[idx])
                bgevents.append(uproot.open(bgfiles[idx]).get('Events'))
        else:
            bgf = uproot.open(bgfiles[fnum])
            bgevents = bgf.get('Events')
        sigf = uproot.open(sigfiles[fnum])
        sigevents = sigf.get('Events')
        
        def loadjets(jets, events,gweights=False):
            jets.eta= pd.DataFrame(events.array('FatJet_eta', executor=executor)).rename(columns=inc)
            jets.phi= pd.DataFrame(events.array('FatJet_phi', executor=executor)).rename(columns=inc)
            jets.pt = pd.DataFrame(events.array('FatJet_pt' , executor=executor)).rename(columns=inc)
            jets.mass=pd.DataFrame(events.array('FatJet_mass', executor=executor)).rename(columns=inc)
            jets.CSVV2 = pd.DataFrame(events.array('FatJet_btagCSVV2', executor=executor)).rename(columns=inc)
            jets.DeepB = pd.DataFrame(events.array('FatJet_btagDeepB', executor=executor)).rename(columns=inc)
            jets.DDBvL = pd.DataFrame(events.array('FatJet_btagDDBvL', executor=executor)).rename(columns=inc)
            jets.msoft = pd.DataFrame(events.array('FatJet_msoftdrop', executor=executor)).rename(columns=inc)
            jets.H4qvs = pd.DataFrame(events.array('FatJet_deepTagMD_H4qvsQCD', executor=executor)).rename(columns=inc)
            jets.n2b1  = pd.DataFrame(events.array('FatJet_n2b1', executor=executor)).rename(columns=inc)
            jets.n2b1[jets.n2b1 < -5] = -2

            jets.event = pd.DataFrame(events.array('event', executor=executor)).rename(columns=inc)
            jets.npvs  = pd.DataFrame(events.array('PV_npvs', executor=executor)).rename(columns=inc)
            jets.npvsG = pd.DataFrame(events.array('PV_npvsGood', executor=executor)).rename(columns=inc)
            
            idxa1 = events.array('FatJet_subJetIdx1')
            idxa2 = events.array('FatJet_subJetIdx2')
            idxa1f = pd.DataFrame(idxa1).rename(columns=inc)
            idxa2f = pd.DataFrame(idxa2).rename(columns=inc)
            submass = events.array('SubJet_mass')
            subtau = events.array('SubJet_tau1')
            jets.submass1 = pd.DataFrame(submass[idxa1[idxa1!=-1]]).rename(columns=inc).add(idxa1f[idxa1f==-1]*0,fill_value=0)
            jets.submass2 = pd.DataFrame(submass[idxa2[idxa2!=-1]]).rename(columns=inc).add(idxa2f[idxa2f==-1]*0,fill_value=0)
            jets.subtau1  = pd.DataFrame(subtau[ idxa1[idxa1!=-1]]).rename(columns=inc).add(idxa1f[idxa1f==-1]*0,fill_value=0)
            jets.subtau2  = pd.DataFrame(subtau[ idxa2[idxa2!=-1]]).rename(columns=inc).add(idxa2f[idxa2f==-1]*0,fill_value=0)
            del idxa1, idxa2, idxa1f, idxa2f, submass, subtau
            
            jets.extweight = jets.event / jets.event
            if gweights:
                jets.extweight[1] = jets.extweight[1] * pd.DataFrame(events.array('Generator_weight'))[0]
                jets.HT = pd.DataFrame(events.array('LHE_HT' , executor=executor)).rename(columns=inc)
            else:
                jets.HT = jets.event * 6000 / jets.event
#            if wname != '':
#                weights = pickle.load(open('weights/'+wname+'-'+fstrip(DATANAME)+'.p',"rb" ))
#                for prop in ['genweights','PUweights','normweights']:
#                    #print('jets.extweight[1]')#,jets.extweight[1])    
#                    jets.extweight[1] = jets.extweight[1] * weights[prop][1]
#            else:
#                jets.extweight = jets.event / jets.event
            for j in range(1,jets.pt.shape[1]):
                jets.event[j+1] = jets.event[1]
                jets.npvs[j+1] = jets.npvs[1]
                jets.npvsG[j+1] = jets.npvsG[1]
                jets.extweight[j+1] = jets.extweight[1]
                jets.HT[j+1] = jets.HT[1]
            return jets
             
        
        if dataflag == True:
            sigjets = loadjets(PhysObj('sigjets'),sigevents)
        else:
            sigjets = loadjets(PhysObj('sigjets'),sigevents,True)
            sigjets.extweight = sigjets.extweight * .0046788#GGH_HPT specific xsec weight
        sigsv = PhysObj('sigsv',sigfiles[fnum],'eta','phi',varname='SV')
            
        if isLHE:
            bgjets = []
            bgsv = []
            for i in range(nlhe):
                bgjets.append(loadjets(PhysObj(f"bgjet{i}"),bgevents[i],True))
                bgsv.append(PhysObj(f"bgsv{i}",bgfiles[fnum*nlhe+i],'eta','phi',varname='SV'))
                
        else:
            if dataflag == -1:
                bgjets = loadjets(PhysObj('bgjets'),bgevents)
            else:
                bgjets = loadjets(PhysObj('bgjets'),bgevents,True)
            bgsv = PhysObj('bgsv',bgfiles[fnum],'eta','phi',varname='SV')

        print(f"Processing {str(len(sigjets.eta))} {Skey} events")
        
        if (not dataflag) and (not LOADMODEL):
            pdgida  = sigevents.array('GenPart_pdgId')
            paridxa = sigevents.array('GenPart_genPartIdxMother')
            parida  = pdgida[paridxa] 
            
            bs = PhysObj('bs')
            
            ## Removes all particles that do not have A parents 
            ## from the GenPart arrays, then removes all particles 
            ## that are not bs after resizing the pdgid array to be a valid mask
            
            bs.oeta = pd.DataFrame(sigevents.array('GenPart_eta')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.ophi = pd.DataFrame(sigevents.array('GenPart_phi')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.opt  = pd.DataFrame(sigevents.array('GenPart_pt' )[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            
            ## Test b order corresponds to As
            testbs = pd.DataFrame(sigevents.array('GenPart_genPartIdxMother')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            ## The first term checks b4 has greater idx than b1, the last two check that the bs are paired
            if ((testbs[4]-testbs[1]).min() <= 0) or ((abs(testbs[2]-testbs[1]) + abs(testbs[4])-testbs[3]).min() != 0):
                print('b to A ordering violated - time to do it the hard way')
                sys.exit()
        
            As = PhysObj('As')    
            As.oeta = pd.DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.ophi = pd.DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.opt =  pd.DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.omass =pd.DataFrame(sigevents.array('GenPart_mass', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            
            higgs = PhysObj('higgs')    
            higgs.eta = pd.DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.phi = pd.DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.pt =  pd.DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            
            slimjets = PhysObj('slimjets')
            slimjets.eta= pd.DataFrame(sigevents.array('Jet_eta', executor=executor)).rename(columns=inc)
            slimjets.phi= pd.DataFrame(sigevents.array('Jet_phi', executor=executor)).rename(columns=inc)
            slimjets.pt = pd.DataFrame(sigevents.array('Jet_pt' , executor=executor)).rename(columns=inc)
            slimjets.mass=pd.DataFrame(sigevents.array('Jet_mass', executor=executor)).rename(columns=inc)
            #sigjets.CSVV2 = pd.DataFrame(sigevents.array('FatJet_btagCSVV2')).rename(columns=inc)
            slimjets.DeepB = pd.DataFrame(sigevents.array('Jet_btagDeepB', executor=executor)).rename(columns=inc)
            #sigjets.DDBvL = pd.DataFrame(sigevents.array('FatJet_btagDDBvL')).rename(columns=inc)
            #sigjets.msoft = pd.DataFrame(sigevents.array('FatJet_msoftdrop')).rename(columns=inc)
            slimjets.DeepFB= pd.DataFrame(sigevents.array('Jet_btagDeepFlavB', executor=executor)).rename(columns=inc)
            slimjets.puid = pd.DataFrame(sigevents.array('Jet_puId', executor=executor)).rename(columns=inc)
        

            ## Figure out how many bs and jets there are
            nb = bs.oeta.shape[1]
            # njet= sigjets.eta.shape[1]
            #nsjet=slimjets.eta.shape[1]
            na = As.oeta.shape[1]
            if na != 2:
                print("More than two As per event, found "+str(na)+", halting")
                sys.exit()
            
            ## Create sorted versions of A values by pt
            for prop in ['eta','phi','pt','mass']:
                As[prop] = pd.DataFrame()
                for i in range(1,3):
                    As[prop][i] = As['o'+prop][As.opt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
                ## Clean up original ordered dataframes; we don't really need them
                #del As['o'+prop]
            
            ## Reorder out b dataframes to match sorted A parents
            tframe = pd.DataFrame()
            tframe[1] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[2] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[3] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            tframe[4] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            for prop in ['eta','phi','pt']:
                bs[prop] = pd.DataFrame()
                bs[prop][1] = bs['o'+prop][tframe][1].dropna().append(bs['o'+prop][tframe][3].dropna()).sort_index()
                bs[prop][2] = bs['o'+prop][tframe][2].dropna().append(bs['o'+prop][tframe][4].dropna()).sort_index()
                bs[prop][3] = bs['o'+prop][~tframe][1].dropna().append(bs['o'+prop][~tframe][3].dropna()).sort_index()
                bs[prop][4] = bs['o'+prop][~tframe][2].dropna().append(bs['o'+prop][~tframe][4].dropna()).sort_index()
                ## Clean up original ordered dataframes; we don't really need them.
                #del bs['o'+prop]
            
#           ## Sort our b dataframes in descending order of pt
#           for prop in ['spt','seta','sphi']:
#               bs[prop] = pd.DataFrame()
#           #bs.spt, bs.seta, bs.sphi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#               for i in range(1,nb+1):
#                   bs[prop][i] = bs[prop[1:]][bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.seta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.sphi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            
#           plots['genAmass'].dfill(As.mass)

            ev = Event(bs,sigjets,As,higgs,sigsv)
        else:
            ev = Event(sigjets,sigsv)
            
        
        if isLHE:
            bev = []
            for i in range(nlhe):
                bev.append(Event(bgjets[i],bgsv[i]))
        else:
            bev = Event(bgjets,bgsv)
        
        if isLHE:
            for jets in bgjets+[sigjets]:
                jets.cut(jets.pt > 170)#240)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#90)#0.25)
                #
                jets.cut(jets.mass > 90)
                jets.cut(jets.msoft < 200)
                jets.cut(jets.npvsG >= 1)
        else:
            for jets in [bgjets, sigjets]:
                jets.cut(jets.pt > 170)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#0.25)
                #
                jets.cut(jets.mass > 90)
                jets.cut(jets.msoft < 200)
                jets.cut(jets.npvsG >= 1)
                


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
            events = sigevents
            event = ev
        elif dataflag == -1:
            events = bgevents
            event = bev
        if dataflag != False:
            ## Apply data cuts
            with open(JSONNAME) as f:
                jdata = json.load(f)
            dtev = PhysObj('event')
            dtev.run = pd.DataFrame(events.array('run')).rename(columns=inc)
            dtev.lb = pd.DataFrame(events.array('luminosityBlock')).rename(columns=inc)
            event.register(dtev)
            event.sync()

            ## Only events that are in keys are kept
            dtev.cut(dtev.run.isin(jdata.keys())==True)
            for elem in dtev:
                dtev[elem]=dtev[elem].astype(int)
            ## Test remaining events for inclusion
            tframe = pd.DataFrame()
            tframe['lb'] = dtev.lb[1]
            tframe['run'] = dtev.run[1]
            def fun(r,lb):
                return any([lb in range(a[0],a[1]+1) for a in jdata[str(r)] ])
            truthframe = pd.DataFrame([fun(r,lb) for r,lb in zip(dtev.run[1].values,dtev.lb[1].values)],index=dtev.run.index,columns=[1])
            dtev.cut(truthframe == True)
            event.sync()

        ## Muon cuts for non-training comparisons
        if (LOADMODEL) and (not passplots):
            muons = PhysObj('Muon',sigfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d')
            muons.ip = abs(muons.dxy / muons.dxyErr)
            ev.register(muons)
            muons.cut(muons.softId > 0.9)
            muons.cut(abs(muons.eta) < 2.4)
            muons.cut(muons.pt > 7)
            muons.cut(muons.ip > 2)
            #muons.cut(muons.ip3d < 0.5)
            ev.sync()
            
            ## Apply these cuts to data events as well
            if isLHE:
                bmuons = []
    #                bmwtpiece = []
                for i in range(nlhe):
                    idx = fnum*nlhe + i
                    bmuons.append(PhysObj('Muon',bgfiles[idx],'softId','eta','pt','dxy','dxyErr','ip3d'))
                    bmuons[i].ip = abs(bmuons[i].dxy / bmuons[i].dxyErr)
                    bev[i].register(bmuons[i])
                    bmuons[i].cut(bmuons[i].softId > 0.9)
                    bmuons[i].cut(abs(bmuons[i].eta) < 2.4)
                    bmuons[i].cut(bmuons[i].pt > 7)
                    bmuons[i].cut(bmuons[i].ip > 2)
                    bmuons[i].cut(bmuons[i].ip3d < 0.5)
                    bev[i].sync()
    
            else:
                bmuons = PhysObj('Muon',bgfiles[fnum],'softId','eta','pt','dxy','dxyErr','ip3d')
                bmuons.ip = abs(bmuons.dxy / bmuons.dxyErr)
                bev.register(bmuons)
                bmuons.cut(muons.softId > 0.9)
                bmuons.cut(abs(muons.eta) < 2.4)
                bmuons.cut(bmuons.pt > 7)
                bmuons.cut(bmuons.ip > 2)
                bmuons.cut(bmuons.ip3d < 0.5)
                bev.sync()

        
        ##########################
        # Training-Specific Cuts #
        ##########################
        if (not dataflag) and (not LOADMODEL):
            jbdr = computedR(sigjets,bs,['jet','b'])
            # ## Create our dR dataframe by populating its first column and naming it accordingly
            # jbdr2 = pd.DataFrame(np.power(sigjets.eta[1]-bs.eta[1],2) + np.power(sigjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            # sjbdr2= pd.DataFrame(np.power(slimjets.eta[1]-bs.eta[1],2) + np.power(slimjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            # ## Loop over jet x b combinations
            # jbstr = []
            # for j in range(1,njet+1):
            #     for b in range(1,nb+1):
            #         ## Make our column name
            #         jbstr.append("Jet "+str(j)+" b "+str(b))
            #         if (j+b==2):
            #             continue
            #         ## Compute and store the dr of the given b and jet for every event at once
            #         jbdr2[jbstr[-1]] = pd.DataFrame(np.power(sigjets.eta[j]-bs.eta[b],2) + np.power(sigjets.phi[j]-bs.phi[b],2))
            #         sjbdr2[jbstr[-1]]= pd.DataFrame(np.power(slimjets.eta[j]-bs.eta[b],2) + np.power(slimjets.phi[j]-bs.phi[b],2))
        
            ## Create a copy array to collapse in jets instead of bs
            blist = []
            # sblist = []
            for b in range(nb):
                blist.append(jbdr.filter(like='b '+str(b+1)))
                blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
                blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))
                # sblist.append(np.sqrt(sjbdr2.filter(like='b '+str(b+1))))
                # sblist[b] = sblist[b][sblist[b].rank(axis=1,method='first') == 1]
                # sblist[b] = sblist[b].rename(columns=lambda x:int(x[4:6]))
        
            ## Trim resolved jet objects        
#           if resjets==3:
#               for i in range(nb):
#                   for j in range(nb):
#                       if i != j:
#                           blist[i] = blist[i][np.logical_not(blist[i] > blist[j])]
#                           blist[i] = blist[i][blist[i]<0.4]
        
            ## Cut our events to only events with 3-4 bs in one fatjet of dR<0.8
            fjets = blist[0][blist[0]<0.8].fillna(0)/blist[0][blist[0]<0.8].fillna(0)
            for i in range(1,4):
                fjets = fjets + blist[i][blist[i]<0.8].fillna(0)/blist[i][blist[i]<0.8].fillna(0)
            fjets = fjets.max(axis=1)
            fjets = fjets[fjets==4].dropna()
            sigjets.trimto(fjets)
            ev.sync()
            
        
        #############################
        # Secondary Vertex Analysis #
        #############################
        ## Compute jet-vertex dR
        sjvdr = computedR(sigjets,sigsv,['jet','sv'])
        sjlist = []
        nsvframe = pd.DataFrame()
        ## Loop over jets
        for j in range(sigjets.eta.shape[1]):
            ## Collect vertex dRs for each jet
            sjlist.append(sjvdr.filter(like=f"jet {j+1}"))
            sjlist[j] = sjlist[j].rename(columns=lambda x:int(x[9:11]))
            ## Remove dR >= 0.8, sum number of remaining vertices
            nsvframe[j+1] = np.sum(sjlist[j][sjlist[j]<0.8].fillna(0)/sjlist[j][sjlist[j]<0.8].fillna(0),axis=1)
        sigjets.nsv = nsvframe
        
        if isLHE:
            for i in range(nlhe):
                if bgjets[i].pt.size <= 0:
                    bgjets[i].nsv = pd.DataFrame()
                    print(f"BG slice {i} had 0 passing events at nsv calc")
                    continue
                bjvdr = computedR(bgjets[i],bgsv[i],['jet','sv'])
                bjlist = []
                nsvframe = pd.DataFrame()
                for j in range(bgjets[i].eta.shape[1]):
                    bjlist.append(bjvdr.filter(like=f"jet {j+1}"))
                    bjlist[j] = bjlist[j].rename(columns=lambda x:int(x[9:11]))
                    nsvframe[j+1] = np.sum(bjlist[j][bjlist[j]<0.8].fillna(0)/bjlist[j][bjlist[j]<0.8].fillna(0),axis=1)
                bgjets[i].nsv = nsvframe
        else:
            bjvdr = computedR(bgjets,bgsv,['jet','sv'])
            bjlist = []
            nsvframe = pd.DataFrame()
            for j in range(bgjets.eta.shape[1]):
                bjlist.append(bjvdr.filter(like=f"jet {j+1}"))
                bjlist[j] = bjlist[j].rename(columns=lambda x:int(x[9:11]))
                nsvframe[j+1] = np.sum(bjlist[j][bjlist[j]<0.8].fillna(0)/bjlist[j][bjlist[j]<0.8].fillna(0),axis=1)
            bgjets.nsv = nsvframe
        
        ##################################
        # Preparing Neural Net Variables #
        ##################################
        
        
        bgjetframe = pd.DataFrame()
        extvars = ['event','extweight','npvs','npvsG']
        if LOADMODEL and (not passplots):
            muvars = ['mpt','meta','mip']
        else: muvars = []
        if isLHE:
            bgpieces = []
            wtpieces = []
            
            for i in range(nlhe):
                tempframe = pd.DataFrame()
                twgtframe = pd.DataFrame()
                for prop in netvars+extvars:
                    twgtframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                if 'eta' in netvars:
                    twgtframe['eta'] = abs(twgtframe['eta'])
                ## Add section for muon variables
                if LOADMODEL and (not passplots):
                    for prop in ['pt','eta','ip']:
                        twgtframe[f"m{prop}"] = bmuons[i][prop][bmuons[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                tempframe = twgtframe.sample(frac=lheweights[i],random_state=6)
                twgtframe['extweight'] = twgtframe['extweight'] * lheweights[i]
                bgpieces.append(tempframe)
                #pickle.dump(tempframe, open(filefix+str(i)+"piece.p", "wb"))
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(bgpieces,ignore_index=True)
            bgrawframe = pd.concat(wtpieces,ignore_index=True)
            bgjetframe = bgjetframe.dropna()
            bgrawframe = bgrawframe.dropna()
            if LOADMODEL and (not passplots):
                bgjetframe['extweight'] = lumipucalc(bgjetframe)
    #            debugframe['extweight'] = lumipucalc(cp.deepcopy(bgjetframe))
    #            sys.exit()
    #            print(bgrawframe['extweight'])
                bgrawframe['extweight'] = lumipucalc(bgrawframe)
    #            print('---->')
    #            print(bgrawframe['extweight'])
            bgjetframe['val'] = 0
            bgrawframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]

        else:
            for prop in netvars + extvars:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            bgjetframe['eta'] = abs(bgjetframe['eta'])
            ## Add section for muon variables
            if LOADMODEL and (not passplots):
                for prop in ['pt','eta','ip']:
                    bgjetframe[f"m{prop}"] = bmuons[prop][bmuons['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
            
                if dataflag != -1:
                    bgjetframe['extweight'] = lumipucalc(bgjetframe)
            bgjetframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        
        nbg = bgtrnframe.shape[0]
            
        sigjetframe = pd.DataFrame()
        for prop in netvars + extvars:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        if 'eta' in netvars:    
            sigjetframe['eta'] = abs(sigjetframe['eta'])   
        ## 
        if dataflag != 1:
            ptwgt = 3.9 - (0.4*np.log2(sigjetframe.pt))
            ptwgt[ptwgt < 0.1] = 0.1
            sigjetframe.extweight = sigjetframe.extweight * ptwgt
        ## Add section for muon variables
        if LOADMODEL and (not passplots):
            for prop in ['pt','eta','ip']:
                sigjetframe[f"m{prop}"] = muons[prop][muons['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
    
            if dataflag != True:
                sigjetframe['extweight'] = lumipucalc(sigjetframe)
        sigjetframe['val'] = 1
        sigtrnframe = sigjetframe[sigjetframe['event']%2 == 0]
        nsig = sigtrnframe.shape[0]
        
        
        print(f"{Skey} cut to {sigjetframe.shape[0]} events")
        print(f"{Bkey} has {bgjetframe.shape[0]} intended events")
            
        extvars = extvars + muvars + ['val']
        
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
            distsf = pd.DataFrame(diststt)
            distsf['W'] = W_inputs[Y_inputs==1].reset_index(drop=True)
            ## Sort both of them together by the confidence values
            distsf = distsf.sort_values(by=[0])
            ## Store the cumulative sum of the weights
            distsf['W_csum'] = distsf.W.cumsum()
            ## Decide what weighted interval each volume should encompass
            isize = distsf.W_csum.max() / 10
            ## label which of those intervals each event falls into
            distsf['bin'] = (distsf.W_csum / isize).apply(math.floor)
            
            for i in range(1,10):
                plots['SensS'][1][i] = distsf[distsf['bin']==i-1][0].max()
                plots['SensB'][1][i] = plots['SensS'][1][i]
            plots['SensS'].fill(diststt,W_inputs[Y_inputs==1])
            plots["SensB"].fill(distbtt,W_inputs[Y_inputs==0])

        elif not passplots:
            hist = pd.DataFrame(history.history)
            #for h in history:
                #hist = pd.concat([hist,pd.DataFrame(h.history)],ignore_index=True)
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
            
        for col in netvars + muvars + ['npvs','npvsG']:
            if not passplots:
                vplots['BG'+col].fill(bgjetframe.reset_index(drop=True)[col],bgjetframe.reset_index(drop=True)['extweight'])
                vplots['SG'+col].fill(sigjetframe.reset_index(drop=True)[col],sigjetframe.reset_index(drop=True)['extweight'])
            else:
                pplots['SG'+col].fill(sigjetframe.reset_index(drop=True)[col],sigjetframe.reset_index(drop=True)['extweight'])
                pplots['SPS'+col].fill(sigjetframe[diststt > passnum].reset_index(drop=True)[col],sigjetframe[diststt > passnum].reset_index(drop=True)['extweight'])
                pplots['SFL'+col].fill(sigjetframe[diststt <= passnum].reset_index(drop=True)[col],sigjetframe[diststt <= passnum].reset_index(drop=True)['extweight'])
                pplots['BG'+col].fill(bgjetframe.reset_index(drop=True)[col],bgjetframe.reset_index(drop=True)['extweight'])
                pplots['BPS'+col].fill(bgjetframe[distbtt > passnum].reset_index(drop=True)[col],bgjetframe[distbtt > passnum].reset_index(drop=True)['extweight'])
                pplots['BFL'+col].fill(bgjetframe[distbtt <= passnum].reset_index(drop=True)[col],bgjetframe[distbtt <= passnum].reset_index(drop=True)['extweight'])

    if False:#LOADMODEL:
        #if gROOT.FindObject('Combined.root'):
         #   rfile = TFile('Combined.root','UPDATE')
        rfile = TFile('Combined.root','UPDATE')
        if dataflag:
            th1 = plots['DistSte'].toTH1('data_obs')
            th12 = plots['DistBte'].toTH1('DnetQCD')
        else:
            th1 = plots['DistSte'].toTH1('SnetSMC')
            th12 = plots['DistBte'].toTH1('SnetQCD')
        rfile.Write()
        rfile.Close()

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
            p.ndivide(sum(p[0]))
        
    if dataflag == True:
        for i in range(len(plots['DistSte'][0])):
            if plots['DistSte'][1][i] >= passnum:
                plots['DistSte'][0][i] = 0
                plots['DistSte'].ser[i] = 0
        
    if LOADMODEL:
        leg = [Sname,Bname]
        Sens = np.sqrt(np.sum(np.power(plots['SensS'][0],2)/plots['SensB'][0]))
        print(f"Calculated Senstivity of {Sens}")
            
        plt.clf()
        plots['DistSte'].make(linestyle='-',**plargs[Skey])
        plots['DistBte'].make(linestyle=':',**plargs[Bkey])
        plots['Distribution'].plot(same=True,legend=leg)
        plt.clf()
        plots['DistSte'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['DistBte'].make(linestyle=':',logv=True,**plargs[Bkey])
        plots['DistributionL'].plot(same=True,logv=True,legend=leg)
        plots['SensS'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['SensB'].title = Sens
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
    

    
    #if POSTWEIGHT:
#    if LOADMODEL:
#        plt.clf()
#        plots['WeightSte'].make(linestyle='-',**plargs[Skey])
#        plots['WeightBte'].make(linestyle=':',**plargs[Bkey])
#        plots['Weights'].plot(same=True,legend=leg)
#    else:
#        plt.clf()
#        plots['WeightStr'].make(color='red',linestyle='-',htype='step')
#        plots['WeightBtr'].make(color='blue',linestyle='-',htype='step')
#        plots['WeightSte'].make(color='red',linestyle=':',htype='step')
#        plots['WeightBte'].make(color='blue',linestyle=':',htype='step')
#        plots['Weights'].plot(same=True,legend=leg)
    

    
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
        if dataflag != True:
            for col in netvars:
                for plot in vplots:
                    vplots[plot].ndivide(sum(abs(vplots[plot][0]+.0001)))
        for col in netvars + muvars + ['npvs','npvsG']:
            plt.clf()
            vplots['SG'+col].make(linestyle='-',**plargs[Skey])
            vplots['BG'+col].make(linestyle=':',**plargs[Bkey],error=dataflag)
            #if POSTWEIGHT:
            #vplots[col].title = 'With post-weighted network training'
            #else:
            #   vplots[col].title = 'With weighted network training'
            vplots[col].plot(same=True,legend=[Sname,Bname])
        
    #if POSTWEIGHT:
    #model.save('postweighted.hdf5')
    #pickle.dump(scaler, open("postweightedscaler.p", "wb"))
    #else:
    if not LOADMODEL:
        model.save('weighted.hdf5')
        pickle.dump(scaler, open("weightedscaler.p", "wb"))
        
    elif not passplots:
        arcdict = {"plots":plots,"vplots":vplots}
        pickle.dump(arcdict, open(plots['Distribution'].fname.split('/')[0]+'/arcdict.p', "wb"))
    #pickle.dump(sigjetframe, open("sigj.p","wb"))

    
        
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