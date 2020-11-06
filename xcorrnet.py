#! /usr/bin/env python

########################################################################
### cross-correlation network analyzer xcorrnet.py                   ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist, PhysObj, Event, inc, fstrip, Hist2d
import pickle
import copy as cp
import math
#from uproot_methods import TLorentzVector, TLorentzVectorArray
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.keras import backend as BE
from keras import backend as K

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
#lheweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
lheweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401,1.0,0.33,0.034,0.034,0.024,0.0024,0.00044]
nlhe = len(lheweights)

DATANAME = '2018D_Parked.root'

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

#%%

def ana(sigfiles,bgfiles,isLHE=False):
    #%%################
    # Plots and Setup #
    ###################
    #training=True
    #training=False
    #tf.random.set_random_seed(2)
    #tf.compat.v1.set_random_seed(2)
    #np.random.seed(2)
    
    #fig = plt.figure(figsize=(10.0,6.0))
    
    global LOADMODEL

    Skey = 'Signal'
    Bkey = 'Background'
    
    l1 = 8
    l2 = 8
    l3 = 8
    alpha = 0.85
    gamma = 0.8
#    bmodel = keras.Sequential([
#            keras.Input(shape=(4,),dtype='float32'),
#            #keras.layers.Flatten(input_shape=(8,)),
#            keras.layers.Dense(l1, activation=tf.nn.relu),
#            keras.layers.Dense(l2, activation=tf.nn.relu),
#            keras.layers.Dense(l3, activation=tf.nn.relu),
#            #keras.layers.Dropout(0.1),
#            keras.layers.Dense(1, activation=tf.nn.sigmoid),
#            ])
#    optimizer  = keras.optimizers.Adam(learning_rate=0.01)
#    bmodel.compile(optimizer=optimizer,#'adam',     
#                  #loss='binary_crossentropy',
#                  #loss=[focal_loss],
#                  #loss=[custom],
#                  loss=[binary_focal_loss(alpha, gamma)],
#                  metrics=['accuracy'])#,tf.keras.metrics.AUC()])
            
    #nbatch = math.floor(nbg / (2*nsig))

    
    netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs']
    bnetvars = ['DeepB','H4qvs','DDBvL','CSVV2']
    pnetvars = ['pt','eta','mass','msoft']
    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
    plots = {
        "SigDist":  Hist2d([20,20],[[0,1],[0,1]],'b-tag network','phys network','netplots/SigDist'),
        "BGDist":   Hist2d([20,20],[[0,1],[0,1]],'b-tag network','phys network','netplots/BGDist'),
        "SigProfile":  Hist(20,(0,1),'b-tag bin','phys network avg','netplots/SigProfile'),
        "BGProfile":   Hist(20,(0,1),'b-tag bin','phys network avg','netplots/BGProfile'),
        "SigRanges":  Hist(20,(0,1),'confidence','events','netplots/SigRanges'),
        "BGRanges":   Hist(20,(0,1),'confidence','events','netplots/BGRanges'),
        }
    for i in range(3):
        plots.update({f"SigRanges{i}":cp.deepcopy(plots['SigRanges']),F"BGRanges{i}":cp.deepcopy(plots['BGRanges'])})
   
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    if isLHE:
        nbg = len(bgfiles)/nlhe
        if float(nbg).is_integer():
            nbg = int(nbg)
        else:
            raise Exception('LHE argument specified, but BG files do not divide evenly into '+str(nlhe))
    else:
        nbg = len(bgfiles)
    nsig = len(sigfiles)
    sigmbg = nbg - nsig
    ## Loop over input files
    for fnum in range(max(nbg,nsig)):
        print('bg',nbg,'sig',nsig)
        
        #####################
        # Loading Variables #
        #####################
        if isLHE:
            print('Opening',sigfiles[fnum],'+ LHE Background')    
        else:
            print('Opening',sigfiles[fnum],'+',bgfiles[fnum])
        
        ## Loop some data if the bg/signal files need to be equalized
        if sigmbg > 0:
            print('Catching up signal')
            sigfiles.append(sigfiles[fnum])
            sigmbg = sigmbg - 1
        elif sigmbg < 0:
            if isLHE:
                print('Catching up background')
                for i in range(nlhe):
                    bgfiles.append(bgfiles[fnum+i])
                    sigmbg = sigmbg + 1
            else:
                print('Catching up background')
                bgfiles.append(bgfiles[fnum])
                sigmbg = sigmbg + 1
        print('diff:',sigmbg)
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
        
        def loadjets(jets, events,wname=''):
            jets.eta= pd.DataFrame(events.array('FatJet_eta', executor=executor)).rename(columns=inc)
            jets.phi= pd.DataFrame(events.array('FatJet_phi', executor=executor)).rename(columns=inc)
            jets.pt = pd.DataFrame(events.array('FatJet_pt' , executor=executor)).rename(columns=inc)
            jets.mass=pd.DataFrame(events.array('FatJet_mass', executor=executor)).rename(columns=inc)
            jets.CSVV2 = pd.DataFrame(events.array('FatJet_btagCSVV2', executor=executor)).rename(columns=inc)
            jets.DeepB = pd.DataFrame(events.array('FatJet_btagDeepB', executor=executor)).rename(columns=inc)
            jets.DDBvL = pd.DataFrame(events.array('FatJet_btagDDBvL', executor=executor)).rename(columns=inc)
            jets.msoft = pd.DataFrame(events.array('FatJet_msoftdrop', executor=executor)).rename(columns=inc)
            jets.H4qvs = pd.DataFrame(events.array('FatJet_deepTagMD_H4qvsQCD', executor=executor)).rename(columns=inc)
            jets.event = pd.DataFrame(events.array('event', executor=executor)).rename(columns=inc)
            jets.npvs  = pd.DataFrame(events.array('PV_npvs', executor=executor)).rename(columns=inc)
            jets.npvsG = pd.DataFrame(events.array('PV_npvsGood', executor=executor)).rename(columns=inc)
            jets.extweight = jets.event / jets.event
            if wname != '':
                weights = pickle.load(open('weights/'+wname+'-'+fstrip(DATANAME)+'.p',"rb" ))
                for prop in ['genweights','PUweights','normweights']:
                    #print('jets.extweight[1]')#,jets.extweight[1])    
                    jets.extweight[1] = jets.extweight[1] * weights[prop][1]
            else:
                jets.extweight = jets.event / jets.event
            for j in range(1,jets.pt.shape[1]):
                jets.event[j+1] = jets.event[1]
                jets.npvs[j+1] = jets.npvs[1]
                jets.npvsG[j+1] = jets.npvsG[1]
                #if POSTWEIGHT:
                jets.extweight[j+1] = jets.extweight[1]
            return jets
        
        sigjets = loadjets(PhysObj('sigjets'),sigevents,fstrip(sigfiles[fnum]))
        
        if isLHE:
            #bgjets = [PhysObj('300'),PhysObj('500'),PhysObj('700'),PhysObj('1000'),PhysObj('1500'),PhysObj('2000'),PhysObj('inf')]
            bgjets = []
            for i in range(nlhe):
                bgjets.append(loadjets(PhysObj(str(i)),bgevents[i],fstrip(bgfiles[(fnum*nlhe)+i])))
        else:
            bgjets = loadjets(PhysObj('bgjets'),bgevents,fstrip(bgfiles[fnum]))

        print(f"Processing {str(len(sigjets.eta))} {Skey} events")
        
        if True:
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
            njet= sigjets.eta.shape[1]
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

            ev = Event(bs,sigjets,As,higgs)
                
        if isLHE:
            for jets in bgjets+[sigjets]:
                jets.cut(jets.pt > 170)#240)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#90)#0.25)
                #
                jets.cut(jets.mass > 90)
        else:
            for jets in [bgjets, sigjets]:
                jets.cut(jets.pt > 170)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#0.25)
                #
                jets.cut(jets.mass > 90)

        bs.cut(bs.pt>5)
        bs.cut(abs(bs.eta)<2.4)
        ev.sync()
        
        slimjets.cut(slimjets.DeepB > 0.1241)
        slimjets.cut(slimjets.DeepFB > 0.277)
        slimjets.cut(slimjets.puid > 0)
        slimjets.trimto(jets.eta)

        
        ##############################
        # Processing and Calculation #
        ##############################

        if True:
            ## Create our dR dataframe by populating its first column and naming it accordingly
            jbdr2 = pd.DataFrame(np.power(sigjets.eta[1]-bs.eta[1],2) + np.power(sigjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            sjbdr2= pd.DataFrame(np.power(slimjets.eta[1]-bs.eta[1],2) + np.power(slimjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            ## Loop over jet x b combinations
            jbstr = []
            for j in range(1,njet+1):
                for b in range(1,nb+1):
                    ## Make our column name
                    jbstr.append("Jet "+str(j)+" b "+str(b))
                    if (j+b==2):
                        continue
                    ## Compute and store the dr of the given b and jet for every event at once
                    jbdr2[jbstr[-1]] = pd.DataFrame(np.power(sigjets.eta[j]-bs.eta[b],2) + np.power(sigjets.phi[j]-bs.phi[b],2))
                    sjbdr2[jbstr[-1]]= pd.DataFrame(np.power(slimjets.eta[j]-bs.eta[b],2) + np.power(slimjets.phi[j]-bs.phi[b],2))
        
            ## Create a copy array to collapse in jets instead of bs
            blist = []
            sblist = []
            for b in range(nb):
                blist.append(np.sqrt(jbdr2.filter(like='b '+str(b+1))))
                blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
                blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))
                sblist.append(np.sqrt(sjbdr2.filter(like='b '+str(b+1))))
                sblist[b] = sblist[b][sblist[b].rank(axis=1,method='first') == 1]
                sblist[b] = sblist[b].rename(columns=lambda x:int(x[4:6]))
        
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
#            fjetsfail = fjets[fjets!=4].dropna()
            fjets = fjets[fjets==4].dropna()
#            ev.sync()
#            sigjetsfail = sigjets.trimto(fjetsfail,split=True)
            sigjets.trimto(fjets)
            ev.sync()

        

        
        ##################################
        # Preparing Neural Net Variables #
        ##################################
        bgjetframe = pd.DataFrame()
        extvars = ['event','extweight','npvs','npvsG']
        
        if isLHE:
            bgpieces = []
            wtpieces = []
            
            for i in range(nlhe):
                tempframe = pd.DataFrame()
                twgtframe = pd.DataFrame()
                for prop in netvars+extvars:
                    twgtframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                if 'eta' in netvars:
                    twgtframe['eta'] = abs(twgtframe['eta'])
                twgtframe['val'] = 0
                tempframe = twgtframe.sample(frac=lheweights[i],random_state=6)
                twgtframe['extweight'] = twgtframe['extweight'] * lheweights[i]
                bgpieces.append(tempframe)
                #pickle.dump(tempframe, open(filefix+str(i)+"piece.p", "wb"))
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(bgpieces,ignore_index=True)
            bgrawframe = pd.concat(wtpieces,ignore_index=True)
            bgjetframe = bgjetframe.dropna()
            bgrawframe = bgrawframe.dropna()
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        else:
            for prop in netvars + extvars:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            bgjetframe['eta'] = abs(bgjetframe['eta'])
            bgjetframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        
        nbg = bgtrnframe.shape[0]
            
        sigjetframe = pd.DataFrame()
        for prop in netvars + extvars:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
        if 'eta' in netvars:    
            sigjetframe['eta'] = abs(sigjetframe['eta'])
        sigjetframe['val'] = 1
        sigtrnframe = sigjetframe[sigjetframe['event']%2 == 0]
        nsig = sigtrnframe.shape[0]
        
#        sigjetfailframe = pd.DataFrame()
#        for prop in netvars + extvars:
#            sigjetfailframe[prop] = sigjetsfail[prop][sigjetsfail['pt'].rank(axis=1,method='first') == 1].max(axis=1)
#        if 'eta' in netvars:    
#            sigjetfailframe['eta'] = abs(sigjetfailframe['eta'])
#        sigjetfailframe['val'] = 1
#        sigtrnFframe = sigjetfailframe[sigjetfailframe['event']%2 == 0]
#        nsig = sigtrnFframe.shape[0]
        
        
        print(f"{Skey} cut to {sigjetframe.shape[0]} events")
        print(f"{Bkey} has {bgjetframe.shape[0]} intended events")
            
        extvars = extvars + ['val']
        #######################
        # Training Neural Net #
        #######################
        

        if isLHE:
            bgjetframe=bgrawframe
                
        X_inputs = pd.concat([bgjetframe,sigjetframe])
#        XF_inputs = pd.concat([bgjetframe,sigjetfailframe])
        W_inputs = X_inputs['extweight']
#        WF_inputs = XF_inputs['extweight']
        Y_inputs = X_inputs['val']
#        YF_inputs = XF_inputs['val']
        X_inputs = X_inputs.drop(extvars,axis=1)
#        XF_inputs = XF_inputs.drop(extvars,axis=1)
        bmodel =    keras.models.load_model('btagfiles/weighted.hdf5', compile=False) 
        physmodel = keras.models.load_model('physfiles/weighted.hdf5', compile=False) 
        bscaler =   pickle.load( open("btagfiles/weightedscaler.p", "rb" ) )
        physcaler = pickle.load( open("physfiles/weightedscaler.p", "rb" ) )
        ##
        #print(scaler.transform(bgpieces[1].drop('val',axis=1)))
        ##
        Xb_inputs   = bscaler.transform(X_inputs.drop(pnetvars,axis=1))
#        XbF_inputs  = bscaler.transform(XF_inputs)
        Xp_inputs   = physcaler.transform(X_inputs.drop(bnetvars,axis=1))
#        XpF_inputs  = physcaler.transform(XF_inputs)

        ##################################
        # Analyzing and Plotting Outputs #
        ##################################
        
        distsb  = bmodel.predict(   Xb_inputs[Y_inputs==1])
        distsp  = physmodel.predict(Xp_inputs[Y_inputs==1])
        
        distbb  = bmodel.predict    (Xb_inputs  [Y_inputs==0])
        distbp  = physmodel.predict (Xp_inputs  [Y_inputs==0])

        plots['SigDist' ].fill(distsb[:,0] ,distsp[:,0] ,weights=W_inputs[Y_inputs==1])
        plots['BGDist'  ].fill(distbb[:,0] ,distbp[:,0] ,weights=W_inputs[Y_inputs==0])
        
        Sprofile, Serr, Bprofile, Berr = [],[],[],[]
        for i in range(plots['SigDist'][0].shape[0]):
            Sprofile.append(np.average(distsp[np.logical_and(
                    distsb > plots['SigDist'][1][i],
                    distsb <= plots['SigDist'][1][i+1])]))
            Serr.append(np.std(distsp[np.logical_and(
                    distsb > plots['SigDist'][1][i],
                    distsb <= plots['SigDist'][1][i+1])]))
            Bprofile.append(np.average(distbp[np.logical_and(
                    distbb > plots['BGDist'][1][i],
                    distbb <= plots['BGDist'][1][i+1])]))
            Berr.append(np.std(distbp[np.logical_and(
                    distbb > plots['BGDist'][1][i],
                    distbb <= plots['BGDist'][1][i+1])]))
        plots['SigProfile'][0] = Sprofile
        plots['SigProfile'].ser = np.power(Serr,2)
        plots['BGProfile'][0] = Bprofile
        plots['BGProfile'].ser = np.power(Berr,2)
        
        sbranges, bbranges = [0.0], [0.0]
        for i in range(1,4):
            sbranges.append(np.sort(distsb.ravel())[math.floor(distsb.shape[0]*i/3)-1])
            bbranges.append(np.sort(distbb.ravel())[math.floor(distbb.shape[0]*i/3)-1])
        
        for i in range(3):
            plots[f"SigRanges{i}"].fill(distsp[np.logical_and(
                    distsb > sbranges[i],
                    distsb <= sbranges[i+1])])
            plots[f"BGRanges{i}"].fill(distbp[np.logical_and(
                    distbb > bbranges[i],
                    distbb <= bbranges[i+1])])
        
        
    for p in [plots['SigDist'],plots['BGDist']]:
        p.plot()
    for p in [plots['SigProfile'],plots['BGProfile']]:
        p.plot(error=True,htype='err')
    plt.clf()
    for plot in ['SigRanges','BGRanges']:
        plots[f"{plot}{0}"].make(linestyle='-',color='b',htype='step')
        plots[f"{plot}{1}"].make(linestyle='--',color='r',htype='step')
        plots[f"{plot}{2}"].make(linestyle=':',color='k',htype='step')
        plots[plot].plot(same=True,legend=['Low Confidence','Medium Confidence','High Confidence'])
        
    #%%

## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        nrgs = len(sys.argv)
        sigfiles = []
        bgfiles = []
        isLHE=False
        ## Check for file sources
        for i in range(nrgs):
            arg = sys.argv[i]
            if '-f' in arg:
                if 's' in arg:
                    fileptr = sigfiles
                elif 'b' in arg:
                    fileptr = bgfiles
                else:
                    dialogue()
                for j in range(i+1,nrgs):
                    if '-' in sys.argv[j]:
                        break
                    else:
                        fileptr.append(sys.argv[j])
                        i = j
            elif '-l' in arg:
                if 's' in arg:
                    fileptr = sigfiles
                elif 'b' in arg:
                    fileptr = bgfiles
                else:
                    dialogue()
                for j in range(i+1,nrgs):
                    if '-' in sys.argv[j]:
                        break
                    else:
                        with open(sys.argv[j],'r') as rfile:
                            for line in rfile:
                                fileptr.append(line.strip('\n'))
                        i = j
            elif '-LHE' in arg:
                isLHE = True
        
        ana(sigfiles,bgfiles,isLHE)

    else:
        dialogue()
        
def dialogue():
    print("Expected:\n xcorrnet.py [-LHE] <-f/-l>s (signal.root) <-f/-l>b (background.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    print("s      Marks the following file(s) as signal")
    print("b      Marks the following file(s) as background")
    print("-LHE   Indicates background files are split by LHE, and should be merged")
    sys.exit(0)
    
if __name__ == "__main__":
    main()

#%%

#X = pd.DataFrame()
#X['x'] = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
#X['y'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#X['z'] = [5,3,4,1,2,4,5,2,3,4,2,4,5,5,5,5,3,4,2,2]
#prop = ['x','y','z']
#Y = pd.DataFrame()
#Y['val']=[1,0,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,0,1]
#y = Y['val']
#
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(8,)),
#    keras.layers.Dense(16, activation=tf.nn.relu),
#    keras.layers.Dense(16, activation=tf.nn.relu),
#    keras.layers.Dense(1, activation=tf.nn.sigmoid),
#])
#
#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#model.fit(X, Y, epochs=150, batch_size=1)
#
#test_loss, test_acc = model.evaluate(X_test, y_test)
#print('Test accuracy:', test_acc)
##%%
#print(model.predict(np.array([[1,1,5],[2,1,4],[3,1,3],[4,1,2],[5,1,1]])))
#print(model.predict(np.array([[1,3,1],[2,5,4],[3,3,3],[4,5,2],[4,3,4]])))
