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
from analib import Hist, PhysObj, Event, inc#, Hist2D
#from uproot_methods import TLorentzVector, TLorentzVectorArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as BE
from keras import backend as K

##Controls how many epochs the network will train for; binary loss will run for a multiple of this
epochs = 50
##The weights LHE segment split data should be merged by
bgweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
nlhe = len(bgweights)
##Switches whether focal loss or binary crossentropy loss is used
FOCAL = True
##Switches tutoring mode on or off
TUTOR = True
TUTOR = False
##Switches whether training statistics are reported or suppressed (for easier to read debugging)
VERBOSE=False

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

def batchtrain(bgtestframe,sigtestframe, scaler):
    l1 = 8
    l2 = 8
    l3 = 8
    lr = 0.01
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(8,)),
            keras.layers.Dense(l1, activation=tf.nn.relu),#, bias_regularizer=tf.keras.regularizers.l2(l=0.0)),
            keras.layers.Dense(l2, activation=tf.nn.relu),
            keras.layers.Dense(l3, activation=tf.nn.relu),
            #keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ])
    optimizer  = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,#'adam',     
                  loss='binary_crossentropy',
                  #loss=[focal_loss],
                  #loss=[custom],
                  #loss=[binary_focal_loss(alpha, gamma)],
                  metrics=['accuracy'])#,tf.keras.metrics.AUC()])
    
    nsig = sigtestframe.shape[0]
    nbg = bgtestframe.shape[0]
    nbatch = math.floor(nbg / (2*nsig))
    #X_test = pd.concat([bgjetframe.drop(bgtestframe.index), sigjetframe.drop(sigtestframe.index)],ignore_index=True)
    #Y_test = X_test['val']
    #X_test = X_test.drop('val',axis=1)
    #X_test = scaler.fit_transform(X_test)
    for i in range(nbatch):
        Xsample= bgtestframe.sample(n=nsig*2,random_state=i)
        X_train = pd.concat([Xsample,sigtestframe.sample(frac=1,random_state=i)],ignore_index=True)
        Y_train= X_train['val']
        X_train = X_train.drop('val',axis=1)
        X_train = scaler.transform(X_train)
        bgtestframe = bgtestframe.drop(Xsample.index) 
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True)
    return history, model
    
    

def tutor(bgjetframe,sigjetframe):
    bgtestframe = bgjetframe.sample(frac=0.7,random_state=6)
    #nbg = bgtestframe.shape[0]
    sigtestframe = sigjetframe.sample(frac=0.7,random_state=6)
    #nsig = sigtestframe.shape[0]

    scaler = MinMaxScaler()
    X_test = pd.concat([bgjetframe.drop(bgtestframe.index), sigjetframe.drop(sigtestframe.index)],ignore_index=True)
    Y_test = X_test['val']
    X_test = X_test.drop('val',axis=1)
    X_test = scaler.fit_transform(X_test)
    
    records = {}
    rsums = {}
    lr=.01
    for l1 in [4,8,16]:
        for l2 in [2,4,8]:
            for l3 in [1,2,4,8]:            
                for alpha in [.7]:
                    for gamma in [.6]:
                        rname = str(l1)+' '+str(l2)+' '+str(l3)+': alpha '+str(alpha)+' gamma '+str(gamma)
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
                                          #loss='binary_crossentropy',
                                          #loss=[focal_loss],
                                          #loss=[custom],
                                          loss=[binary_focal_loss(alpha, gamma)],
                                          metrics=['accuracy'])#,tf.keras.metrics.AUC()])

                            X_train = pd.concat([bgtestframe,sigtestframe],ignore_index=True)
                            Y_train= X_train['val']
                            X_train = X_train.drop('val',axis=1)
                            X_train = scaler.transform(X_train)
                            model.fit(X_train, Y_train, epochs=25, batch_size=5128,shuffle=True)

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

def ana(sigfiles,bgfiles,isLHE=False):
    #%%################
    # Plots and Setup #
    ###################
    #training=True
    #training=False
    #tf.random.set_random_seed(2)
    #tf.compat.v1.set_random_seed(2)
    #np.random.seed(2)
    
    l1 = 8
    l2 = 8
    l3 = 8
    alpha = 0.7
    gamma = 0.6
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(8,)),
            keras.layers.Dense(l1, activation=tf.nn.relu),#, bias_regularizer=tf.keras.regularizers.l2(l=0.0)),
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
    
    netvars = ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']
    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
    plots = {
        "Distribution": Hist(20,(0,1),'Signal (Red) and Background (Blue) testing (..) and training samples','% of Events','netplots/Distribution'),
        "DistributionL": Hist(20,(0,1),'Signal (Red) and Background (Blue) testing (..) and training samples','% of Events','netplots/LogDistribution'),
        "DistStr":  Hist(20,(0,1)),
        "DistSte":  Hist(20,(0,1)),
        "DistBtr":  Hist(20,(0,1)),
        "DistBte":  Hist(20,(0,1)),
        "LossvEpoch":   Hist(epochs,(0.5,epochs+.5),'Epoch Number','Loss','netplots/LossvEpoch'),
        "AccvEpoch":Hist(epochs,(0.5,epochs+.5),'Epoch Number','Accuracy','netplots/AccvEpoch'),
    }
    vplots = {
        "pt":       Hist(80 ,(150,550)  ,'pT for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/pt'),
        "BGpt":     Hist(80 ,(150,550)),
        "SGpt":     Hist(80 ,(150,550)),
        "RWpt":     Hist(80 ,(150,550)),
        "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/eta'),
        "BGeta":    Hist(15 ,(0,3)),
        "SGeta":    Hist(15 ,(0,3)),
        "RWeta":    Hist(15 ,(0,3)),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/phi'),
        "BGphi":    Hist(32 ,(-3.2,3.2)),
        "SGphi":    Hist(32 ,(-3.2,3.2)),
        "RWphi":    Hist(32 ,(-3.2,3.2)),
        "mass":     Hist(50 ,(0,200)    ,'mass for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/mass'),
        "BGmass":   Hist(50 ,(0,200)),
        "SGmass":   Hist(50 ,(0,200)),
        "RWmass":   Hist(50 ,(0,200)),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/CSVV2'),
        "BGCSVV2":  Hist(22 ,(0,1.1)),
        "SGCSVV2":  Hist(22 ,(0,1.1)),
        "RWCSVV2":  Hist(22 ,(0,1.1)),
        "DeepB":    Hist(22 ,(0,1.1)    ,'DeepB for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/DeepB'),
        "BGDeepB":  Hist(22 ,(0,1.1)),
        "SGDeepB":  Hist(22 ,(0,1.1)),
        "RWDeepB":  Hist(22 ,(0,1.1)),
        "msoft":    Hist(50 ,(0,200)    ,'msoft for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/msoft'),
        "BGmsoft":  Hist(50 ,(0,200)),
        "SGmsoft":  Hist(50 ,(0,200)),
        "RWmsoft":  Hist(50 ,(0,200)),
        "DDBvL":    Hist(22 ,(0,1.1)    ,'DDBvL for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/DDBvL'),
        "BGDDBvL":  Hist(22 ,(0,1.1)),
        "SGDDBvL":  Hist(22 ,(0,1.1)),
        "RWDDBvL":  Hist(22 ,(0,1.1)),
        #"LHEHT":    Hist(400,(0,4000)   ,'LHE_HT for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/LHE_HT'),
        #"BGLHEHT":  Hist(400,(0,4000)),
        #"SGLHEHT":  Hist(400,(0,4000)),
        #"DTLHEHT":  Hist(400,(0,4000)),
    }
    if isLHE:
        lheplots = {}
        for i in range(nlhe):
            lheplots.update({'dist'+str(i):Hist(20,(0,1),'Normalized MC background classifcation','% of Events','netplots/LHEdist_'+str(i)),})
            lheplots['dist'+str(i)].title = 'Distrubution for LHE segment '+str(i)
#    for plot in plots:
#        plots[plot].title = files[0]

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
        
        
        if isLHE:
            bgjets = [PhysObj('300'),PhysObj('500'),PhysObj('700'),PhysObj('1000'),PhysObj('1500'),PhysObj('2000'),PhysObj('inf')]
            for i in range(nlhe):
                bgjets[i].eta= np.abs(pd.DataFrame(bgevents[i].array('FatJet_eta')).rename(columns=inc))
                bgjets[i].phi= pd.DataFrame(bgevents[i].array('FatJet_phi')).rename(columns=inc)
                bgjets[i].pt = pd.DataFrame(bgevents[i].array('FatJet_pt')).rename(columns=inc)
                bgjets[i].mass=pd.DataFrame(bgevents[i].array('FatJet_mass')).rename(columns=inc)
                bgjets[i].CSVV2 = pd.DataFrame(bgevents[i].array('FatJet_btagCSVV2')).rename(columns=inc)
                bgjets[i].DeepB = pd.DataFrame(bgevents[i].array('FatJet_btagDeepB')).rename(columns=inc)
                bgjets[i].DDBvL = pd.DataFrame(bgevents[i].array('FatJet_btagDDBvL')).rename(columns=inc)
                bgjets[i].msoft = pd.DataFrame(bgevents[i].array('FatJet_msoftdrop')).rename(columns=inc)
                #bgjets[i].LHEHT = pd.DataFrame(bgevents[i].array('LHE_HT')).rename(columns=inc)
        else:
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
        
        if isLHE:
            for jets in bgjets+[sigjets]:
                jets.cut(jets.pt>170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 0.25)
        else:
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
        
        if isLHE:
            bgpieces = []
            wtpieces = []
            for i in range(nlhe):
                tempframe = pd.DataFrame()
                twgtframe = pd.DataFrame()
                for prop in netvars:
                    tempframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                twgtframe = tempframe.sample(frac=bgweights[i],random_state=6)
                #tempframe = tempframe[tempframe != 0]
                tempframe['val'] = 0
                twgtframe['val'] = 0
                bgpieces.append(tempframe)
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(wtpieces,ignore_index=True)
            bgrawframe = pd.concat(bgpieces,ignore_index=True)
            bgjetframe = bgjetframe.dropna()
            bgrawframe = bgrawframe.dropna()
        else:
            for prop in netvars:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                bgjetframe['val'] = 0
        bgtestframe = bgjetframe.sample(frac=0.7,random_state=6)
        rwtestframe = bgrawframe.sample(frac=0.7,random_state=6)
        nbg = bgtestframe.shape[0]
            
        sigjetframe = pd.DataFrame()
        for prop in netvars:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            sigjetframe['val'] = 1
        sigtestframe = sigjetframe.sample(frac=0.7,random_state=6)
        nsig = sigtestframe.shape[0]
        
        print('Signal cut to ',sigjetframe.shape[0], ' events')
        print('Background has ',bgjetframe.shape[0],' events')
        
        if TUTOR == True:
            tutor(bgjetframe,sigjetframe)
        else:
            X_test = pd.concat([bgjetframe.drop(bgtestframe.index), sigjetframe.drop(sigtestframe.index)],ignore_index=True)
            Y_test = X_test['val']
            X_test = X_test.drop('val',axis=1)
            X_test = scaler.fit_transform(X_test)
            ## toggle between rwtestframe and bgtestframe to control whether weights are used to train
            X_train = pd.concat([rwtestframe,sigtestframe],ignore_index=True)
            Y_train= X_train['val']
            X_train = X_train.drop('val',axis=1)
            X_train = scaler.transform(X_train)
            
            if FOCAL:
                history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True,verbose=VERBOSE)
            else:
                history, model = batchtrain(bgtestframe,sigtestframe,scaler)

        rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
        trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train).ravel())
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
        
        
        diststr = model.predict(X_train[Y_train==1])
        distste = model.predict(X_test[Y_test==1])
        distbtr = model.predict(X_train[Y_train==0])
        distbte = model.predict(X_test[Y_test==0])
        
        if isLHE:
            for i in range(nlhe):
                test = pd.concat(wtpieces,ignore_index=True).drop('val',axis=1)
                piece = wtpieces[i].drop('val',axis=1)
                piece = piece.reset_index(drop=True)
                #print(piece)
                #print('xxxxx')
                piece = scaler.transform(test)
                print(piece)
                #print('............')
                print(X_test[Y_test==0])
                lhedist = model.predict(piece)
                #print(lhedist)
                lheplots['dist'+str(i)].fill(lhedist)
        
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
        plt.clf()
        plots['LossvEpoch'].plot()
        plots['AccvEpoch'].plot()
        plt.clf()
        
        plots['DistStr'].fill(diststr)
        plots['DistSte'].fill(distste)
        plots['DistBtr'].fill(distbtr)
        plots['DistBte'].fill(distbte)
        plt.clf()
        
        for col in netvars:
            vplots['BG'+col].fill(bgjetframe[col])
            vplots['SG'+col].fill(sigjetframe[col])
            vplots['RW'+col].fill(bgrawframe[col])

        
        
        plt.clf()
        plt.plot([0,1],[0,1],'k--')
        plt.plot(rocx,rocy,'red')
        plt.plot(trocx,trocy,'b:')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(['y=x','Validation','Training'])
        plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
        plt.savefig('netplots/ROC_'+str(fnum))
    
    for p in [plots['DistStr'],plots['DistSte'],plots['DistBtr'],plots['DistBte']]:
        #p.norm(sum(p[0]))
        p[0] = p[0]/sum(p[0])    
    plt.clf()
    plots['DistStr'].make(color='red',linestyle='-',htype='step')
    plots['DistBtr'].make(color='blue',linestyle='-',htype='step')
    plots['DistSte'].make(color='red',linestyle=':',htype='step')
    plots['DistBte'].make(color='blue',linestyle=':',htype='step')
    plots['Distribution'].plot(same=True)
    
    plt.clf()
    plots['DistStr'].make(color='red',linestyle='-',htype='step',logv=True)
    plots['DistBtr'].make(color='blue',linestyle='-',htype='step',logv=True)
    plots['DistSte'].make(color='red',linestyle=':',htype='step',logv=True)
    plots['DistBte'].make(color='blue',linestyle=':',htype='step',logv=True)
    plots['DistributionL'].plot(same=True,logv=True)
    
    for col in netvars:
        vplots['BG'+col][0] = vplots['BG'+col][0]/sum(vplots['BG'+col][0])
        vplots['SG'+col][0] = vplots['SG'+col][0]/sum(vplots['SG'+col][0])
        vplots['RW'+col][0] = vplots['RW'+col][0]/sum(vplots['RW'+col][0])

    if isLHE:
        for i in range(nlhe):
            lheplots['dist'+str(i)][0] = lheplots['dist'+str(i)][0]/sum(lheplots['dist'+str(i)][0])
            lheplots['dist'+str(i)].plot(htype='step',logv=True)
            
    for col in netvars:
        plt.clf()
        vplots['SG'+col].make(color='red'  ,linestyle='-',htype='step')
        vplots['RW'+col].make(color='black',linestyle='--',htype='step')
        vplots['BG'+col].make(color='blue' ,linestyle=':',htype='step')
        vplots[col].plot(same=True)
    
        
    #%%
    return auc(rocx,rocy)
        #sys.exit()

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
    print("Expected mndwrm.py [-LHE] <-f/-l>s (signal.root) <-f/-l>b (background.root)")
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
