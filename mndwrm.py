#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

#import ROOT as R
#R.gROOT.SetBatch(True)  ## Don't display histograms or canvases when drawn

#import os
#import subprocess
import sys
import uproot
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist, PhysObj, Event, Hist2d, inc
from uproot_methods import TLorentzVector, TLorentzVectorArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as BE
from keras import backend as K

def binary_focal_loss(gamma=2., alpha=.25):
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

def custom(y_true,y_pred):
    gamma = 2
    alpha = .25
    
    #alphaT = np.where(y_true==1,y_true*alpha,(1-alpha)*np.ones_like(y_true))
    #pT = np.where(y_true==1,y_pred,(1-y_pred)*np.ones_like(y_pred))
    #return np.sum((-1*alphaT) * np.power((1 - pT),gamma) * np.log(pT)

    alphaT = tf.where(tf.equal(y_true,1),y_true*alpha,(1-alpha)*tf.ones_like(y_true))
    pT = tf.where(tf.equal(y_true,1),y_pred,(1-y_pred)*tf.ones_like(y_true))
    pT = BE.clip(pT,BE.epsilon(),1-BE.epsilon())
    loss = tf.reduce_mean((-1*alphaT) * tf.pow((1 - pT),gamma) * BE.log(pT))
    return loss
    #return tf.reduce_mean((-1*alphaT) * tf.pow((1 - pT),gamma) * tf.log(pT))

def focal_loss(yTrue, yGuess):
    gamma = .2
    alpha = .85
    pt1 = tf.where(tf.equal(yTrue, 1), yGuess, tf.ones_like(yGuess))
    pt0 = tf.where(tf.equal(yTrue, 0), yGuess, tf.zeros_like(yGuess))
    return -BE.sum(alpha * BE.pow(1. - pt1, gamma) * BE.log(pt1))-BE.sum((1-alpha) * BE.pow( pt0, gamma) * BE.log(1. - pt0))

def tutor(bgjetframe,sigjetframe):
    bgtestframe = bgjetframe.sample(frac=0.7,random_state=6)
    nbg = bgtestframe.shape[0]
    sigtestframe = sigjetframe.sample(frac=0.7,random_state=6)
    nsig = sigtestframe.shape[0]
    nbatch = math.floor(nbg / (2*nsig))
    bgfrac = nbatch/nbg
    X_test = pd.concat([bgjetframe.drop(bgtestframe.index), sigjetframe.drop(sigtestframe.index)],ignore_index=True)
    Y_test = X_test['val']
    X_test = X_test.drop('val',axis=1)
    
    records = {}
    rsums = {}
    for l1 in [8]:
        for l2 in [8]:
            for l3 in [8]:            
                for lr in [0.1,0.05,0.01,0.005,0.001]:
                    for placeholder in [1]:
                        rname = str(l1)+' '+str(l2)+' '+str(l3)+': lr '+str(lr)
                        aoc = []
                        for i in range(10):
                            #tf.random.set_random_seed(2)
                            #tf.compat.v1.random.set_random_seed(2)
                            #tf.compat.v1.set_random_seed(2)
                            np.random.seed(2)
                            model = keras.Sequential([
                                    keras.layers.Flatten(input_shape=(8,)),
                                    keras.layers.Dense(l1, activation=tf.nn.relu),
                                    keras.layers.Dense(l2, activation=tf.nn.relu),
                                    keras.layers.Dense(l3, activation=tf.nn.relu),
                                    keras.layers.Dense(1, activation=tf.nn.sigmoid),
                                    ])
                            optimizer  = keras.optimizers.Adam(learning_rate=lr)
                            model.compile(optimizer=optimizer,     
                                          loss='binary_crossentropy',
                                          #loss=[focal_loss],
                                          #loss=[custom],
                                          #loss=[binary_focal_loss(alpha, gamma)],
                                          metrics=['accuracy'])#,tf.keras.metrics.AUC()])
                            bgtmpframe = bgtestframe
                            for i in range(nbatch):
                                #print(bgtestframe.shape[0],nsig)
                                Xsample= bgtmpframe.sample(n=nsig*2,random_state=i)
                                X_train = pd.concat([Xsample,sigjetframe.sample(frac=1,random_state=i)],ignore_index=True)
                                Y_train= X_train['val']
                                X_train = X_train.drop('val',axis=1)
                                bgtmpframe = bgtmpframe.drop(Xsample.index) 
                                model.fit(X_train, Y_train, epochs=10, batch_size=5128,shuffle=True)
                            #model.fit(X_train, Y_train, epochs=25, batch_size=5128,shuffle=True)
                            rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
                            aoc.append(auc(rocx,rocy))
                        aocsum = np.sum(aoc)/10               
                        records[rname] = aocsum
                        rsums[rname] = aoc
                        print(rname,' = ',aocsum)
                    
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
#%%

def ana(sigfiles,bgfiles):
    #%%################
    # Plots and Setup #
    ###################
    training=True
    training=False
    #tf.random.set_random_seed(2)
    #tf.compat.v1.set_random_seed(2)
    #np.random.seed(2)

    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
    plots = {
        "Distribution": Hist(20,(0,1),'Signal (Red) and Background (Blue) testing (..) and training samples','Events','netplots/distribution'),
        "DistStr":  Hist(20,(0,1)),
        "DistSte":  Hist(20,(0,1)),
        "DistBtr":  Hist(20,(0,1)),
        "DistBte":  Hist(20,(0,1)),
        "LossvEpoch":   Hist(25,(0.5,25.5),'Epoch Number','Loss','netplots/LossvEpoch'),
        "AccvEpoch":Hist(25,(0.5,25.5),'Epoch Number','Accuracy','netplots/AccvEpoch'),
    }
#    for plot in plots:
#        plots[plot].title = files[0]

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    nbg = len(bgfiles)
    nsig = len(sigfiles)
    sigmbg = nbg - nsig
    ## Loop over input files
    for fnum in range(max(nbg,nsig)):
        
        #####################
        # Loading Variables #
        #####################
        print('Opening ',sigfiles[fnum],' + ',bgfiles[fnum])
        
        ## Loop some data if the bg/signal files need to be equalized
        if sigmbg > 0:
            print('Catching up signal')
            sigfiles.append(sigfiles[fnum])
            sigmbg = sigmbg - 1
        elif sigmbg < 0:
            print('Catching up background')
            bgfiles.append(bgfiles[fnum])
            sigmbg = sigmbg + 1
        
        ## Open our file and grab the events tree
        sigf = uproot.open(sigfiles[fnum])#'nobias.root')
        bgf = uproot.open(bgfiles[fnum])
        sigevents = sigf.get('Events')
        bgevents = bgf.get('Events')


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
        
        As.oeta = pd.DataFrame(sigevents.array('GenPart_eta')[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        As.ophi = pd.DataFrame(sigevents.array('GenPart_phi')[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        As.opt =  pd.DataFrame(sigevents.array('GenPart_pt' )[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        As.omass =pd.DataFrame(sigevents.array('GenPart_mass')[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        
        higgs = PhysObj('higgs')
        
        higgs.eta = pd.DataFrame(sigevents.array('GenPart_eta')[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        higgs.phi = pd.DataFrame(sigevents.array('GenPart_phi')[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        higgs.pt =  pd.DataFrame(sigevents.array('GenPart_pt' )[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        
        
        sigjets = PhysObj('sigjets')

        sigjets.eta= pd.DataFrame(sigevents.array('FatJet_eta')).rename(columns=inc)
        sigjets.phi= pd.DataFrame(sigevents.array('FatJet_phi')).rename(columns=inc)
        sigjets.pt = pd.DataFrame(sigevents.array('FatJet_pt')).rename(columns=inc)
        sigjets.mass=pd.DataFrame(sigevents.array('FatJet_mass')).rename(columns=inc)
        sigjets.CSVV2 = pd.DataFrame(sigevents.array('FatJet_btagCSVV2')).rename(columns=inc)
        sigjets.DeepB = pd.DataFrame(sigevents.array('FatJet_btagDeepB')).rename(columns=inc)
        sigjets.DDBvL = pd.DataFrame(sigevents.array('FatJet_btagDDBvL')).rename(columns=inc)
        sigjets.msoft = pd.DataFrame(sigevents.array('FatJet_msoftdrop')).rename(columns=inc)
        
        slimjets = PhysObj('slimjets')
        slimjets.eta= pd.DataFrame(sigevents.array('Jet_eta')).rename(columns=inc)
        slimjets.phi= pd.DataFrame(sigevents.array('Jet_phi')).rename(columns=inc)
        slimjets.pt = pd.DataFrame(sigevents.array('Jet_pt')).rename(columns=inc)
        slimjets.mass=pd.DataFrame(sigevents.array('Jet_mass')).rename(columns=inc)
        #sigjets.CSVV2 = pd.DataFrame(sigevents.array('FatJet_btagCSVV2')).rename(columns=inc)
        slimjets.DeepB = pd.DataFrame(sigevents.array('Jet_btagDeepB')).rename(columns=inc)
        #sigjets.DDBvL = pd.DataFrame(sigevents.array('FatJet_btagDDBvL')).rename(columns=inc)
        #sigjets.msoft = pd.DataFrame(sigevents.array('FatJet_msoftdrop')).rename(columns=inc)
        slimjets.DeepFB= pd.DataFrame(sigevents.array('Jet_btagDeepFlavB')).rename(columns=inc)
        slimjets.puid = pd.DataFrame(sigevents.array('Jet_puId')).rename(columns=inc)
        
        bgjets = PhysObj('bgjets')
        
        bgjets.eta= pd.DataFrame(bgevents.array('FatJet_eta')).rename(columns=inc)
        bgjets.phi= pd.DataFrame(bgevents.array('FatJet_phi')).rename(columns=inc)
        bgjets.pt = pd.DataFrame(bgevents.array('FatJet_pt')).rename(columns=inc)
        bgjets.mass=pd.DataFrame(bgevents.array('FatJet_mass')).rename(columns=inc)
        bgjets.CSVV2 = pd.DataFrame(bgevents.array('FatJet_btagCSVV2')).rename(columns=inc)
        bgjets.DeepB = pd.DataFrame(bgevents.array('FatJet_btagDeepB')).rename(columns=inc)
        bgjets.DDBvL = pd.DataFrame(bgevents.array('FatJet_btagDDBvL')).rename(columns=inc)
        bgjets.msoft = pd.DataFrame(bgevents.array('FatJet_msoftdrop')).rename(columns=inc)
        #bgjets.DeepFB= pd.DataFrame(bgevents.array('Jet_btagDeepFlavB')).rename(columns=inc)

        print('Processing ' + str(len(bs.oeta)) + ' events')

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
            
#        ## Sort our b dataframes in descending order of pt
#        for prop in ['spt','seta','sphi']:
#            bs[prop] = pd.DataFrame()
#        #bs.spt, bs.seta, bs.sphi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#            for i in range(1,nb+1):
#                bs[prop][i] = bs[prop[1:]][bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#            #bs.seta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#            #bs.sphi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            
#        plots['genAmass'].dfill(As.mass)

        ev = Event(bs,sigjets,As,higgs)
        
        for jets in [sigjets,bgjets]:
            jets.cut(jets.pt>170)
            jets.cut(abs(jets.eta)<2.4)
            jets.cut(jets.DDBvL > 0.6)
            jets.cut(jets.DeepB > 0.4184)
            jets.cut(jets.msoft > 0.25)

        bs.cut(bs.pt>5)
        bs.cut(abs(bs.eta)<2.4)
        ev.sync()
        
        slimjets.cut(slimjets.DeepB > 0.1241)
        slimjets.cut(slimjets.DeepFB > 0.277)
        slimjets.cut(slimjets.puid > 0)
        slimjets.trimTo(jets.eta)
        
        ##############################
        # Processing and Calculation #
        ##############################

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
#        if resjets==3:
#            for i in range(nb):
#                for j in range(nb):
#                    if i != j:
#                        blist[i] = blist[i][np.logical_not(blist[i] > blist[j])]
#                        blist[i] = blist[i][blist[i]<0.4]
        
        ## Cut our events to only events with 3-4 bs in one fatjet of dR<0.8
        fjets = blist[0][blist[0]<0.8].fillna(0)/blist[0][blist[0]<0.8].fillna(0)
        for i in range(1,4):
            fjets = fjets + blist[i][blist[i]<0.8].fillna(0)/blist[i][blist[i]<0.8].fillna(0)
        fjets = fjets.max(axis=1)
        fjets = fjets[fjets==4].dropna()
        sigjets.trimTo(fjets)
        ev.sync()
        

        
        #######################
        # Training Neural Net #
        #######################
        bgjetframe = pd.DataFrame()
        
        for prop in ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']:
            bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            bgjetframe['val'] = 0
        bgtestframe = bgjetframe.sample(frac=0.7,random_state=6)
        sigjetframe = pd.DataFrame()
        nbg = bgtestframe.shape[0]
        for prop in ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            sigjetframe['val'] = 1
        sigtestframe = sigjetframe.sample(frac=0.7,random_state=6)
        nsig = sigtestframe.shape[0]
        
        print('Signal cut to ',sigjetframe.shape[0], ' events')
        print('Background has ',bgjetframe.shape[0],' events')
        
        if training == True:
            tutor(bgjetframe,sigjetframe)
        else:
            l1 = 8
            l2 = 8
            l3 = 8
            
            model = keras.Sequential([
                    keras.layers.Flatten(input_shape=(8,)),
                    keras.layers.Dense(l1, activation=tf.nn.relu),#, bias_regularizer=tf.keras.regularizers.l2(l=0.0)),
                    keras.layers.Dense(l2, activation=tf.nn.relu),
                    keras.layers.Dense(l3, activation=tf.nn.relu),
                    #keras.layers.Dropout(0.1),
                    keras.layers.Dense(1, activation=tf.nn.sigmoid),
                    ])
            optimizer  = keras.optimizers.Adam(learning_rate=0.005)
            model.compile(optimizer=optimizer,#'adam',     
                         loss='binary_crossentropy',
                         #loss=[focal_loss],
                         #loss=[custom],
                         #loss=[binary_focal_loss(alpha, gamma)],
                         metrics=['accuracy'])#,tf.keras.metrics.AUC()])
            
            nbatch = math.floor(nbg / (2*nsig))
            bgfrac = nbatch/nbg
            X_test = pd.concat([bgjetframe.drop(bgtestframe.index), sigjetframe.drop(sigtestframe.index)],ignore_index=True)
            Y_test = X_test['val']
            X_test = X_test.drop('val',axis=1)
            
            history = {}
            for i in range(nbatch):
                Xsample= bgtestframe.sample(n=nsig*2,random_state=i)
                X_train = pd.concat([Xsample,sigjetframe.sample(frac=1,random_state=i)],ignore_index=True)
                Y_train= X_train['val']
                X_train = X_train.drop('val',axis=1)
                bgtestframe = bgtestframe.drop(Xsample.index) 
                history = model.fit(X_train, Y_train, epochs=10, batch_size=5128,shuffle=True)
                #history.update(htmp)
            
        rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
        trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train).ravel())
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
        
        diststr = model.predict(X_train[Y_train==1])
        distste = model.predict(X_test[Y_test==1])
        distbtr = model.predict(X_train[Y_train==0])
        distbte = model.predict(X_test[Y_test==0])
        
        hist = pd.DataFrame(history.history)
        #for h in history:
            #hist = pd.concat([hist,pd.DataFrame(h.history)],ignore_index=True)
        hist['epoch'] = history.epoch
        #plots['LossvEpoch'][0]=hist['loss']
        #plots['AccvEpoch'][0]=hist['acc']
        #plots['LossvEpoch'][0]=hist['epoch']
        #plots['AccvEpoch'][0]=hist['epoch']
        plt.clf()
        #plots['LossvEpoch'].plot()
        #plots['AccvEpoch'].plot()
        plt.clf()
        
        plots['DistStr'].fill(diststr)
        plots['DistSte'].fill(distste)
        plots['DistBtr'].fill(distbtr)
        plots['DistBte'].fill(distbte)
        plt.clf()
        for p in [plots['DistStr'],plots['DistSte'],plots['DistBtr'],plots['DistBte']]:
            #p.norm(sum(p[0]))
            p[0] = p[0]/sum(p[0])
        plots['DistStr'].make(color='red',linestyle='-',htype='step')
        plots['DistBtr'].make(color='blue',linestyle='-',htype='step')
        plots['DistSte'].make(color='red',linestyle=':',htype='step')
        plots['DistBte'].make(color='blue',linestyle=':',htype='step')
        plots['Distribution'].plot(same=True)
        

        plt.clf()
        plt.plot([0,1],[0,1],'k--')
        plt.plot(rocx,rocy,'red')
        plt.plot(trocx,trocy,'b:')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(['y=x','Validation','Training'])
        plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
        plt.savefig('netplots/ROC')
        
        #%%
        return auc(rocx,rocy)
        #sys.exit()

    #%%
    
## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        sigfiles = []
        bgfiles = []
        ## Check for file sources
        if '-f' in sys.argv:
            idx = sys.argv.index('-f')+1
            try:                
                for i in sys.argv[idx:]:
                    if i == '-s':
                        fileptr = sigfiles
                    elif i == '-b':
                        fileptr = bgfiles
                    else:
                        fileptr.append(i)
            except:
                dialogue()
        elif '-l' in sys.argv:
            idx = sys.argv.index('l')+1
            try:
                for i in sys.argv[idx:]:
                    if i == '-s':
                        fileptr = sigfiles
                    elif i == '-b':
                        fileptr = bgfiles
                    else:
                        with open(sys.argv[3],'r') as rfile:
                            for line in rfile:
                                fileptr.append(line.strip('\n')) 
            except:
                dialogue()
        else:
            dialogue()
        ## Check specified run mode
        ana(sigfiles,bgfiles)
    else:
        dialogue()
        
def dialogue():
    print("Expected mndwrm.py <-f/-l> -s (signal.root) -b (background.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
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
