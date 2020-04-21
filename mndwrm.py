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
#%%

def ana(sigfiles,bgfiles):
    #%%################
    # Plots and Setup #
    ###################
    
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(8,)),
            keras.layers.Dense(8, activation=tf.nn.relu),
            keras.layers.Dense(4, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    
    model.compile(optimizer='adam',     
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    ## Define what pdgId we expect the A to have
    #Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
#    bjplots = {}
#    for i in range(1,5):
#        bjplots.update({
#        "s_beta"+str(i):      Hist(33 ,(-3.3,3.3)   ,'GEN b '+str(i)+' Eta (ranked by pT)','Events','upplots/s_beta'+str(i)),
#        "s_bpT"+str(i):       Hist(60 ,(0,120)      ,'GEN pT of b '+str(i)+' (ranked by pT)','Events','upplots/s_bpT'+str(i)),
#        "s_bjetpT"+str(i):    Hist(60 ,(0,120)      ,'Matched RECO jet '+str(i)+' pT (ranked by b pT)','Events','upplots/s_RjetpT'+str(i)),
#        "s_bjeteta"+str(i):   Hist(33 ,(-3.3,3.3)   ,'Matched RECO jet '+str(i)+' Eta (ranked by b pT)','Events','upplots/s_Rjeteta'+str(i)),
#        "s_bjdR"+str(i):      Hist(90 ,(0,3)        ,'GEN b '+str(i)+' (ranked by pT) to matched jet dR','Events','upplots/s_bjdR'+str(i))
#        })
    plots = {
        "Distribution": Hist(100,(0,1),'Signal (Red) and Background (Blue) training (..) and test samples','Events','netplots/distribution'),
        "DistStr":  Hist(100,(0,1)),
        "DistSte":  Hist(100,(0,1)),
        "DistBtr":  Hist(100,(0,1)),
        "DistBte":  Hist(100,(0,1))
#        "HpT":      Hist(60 ,(0,320)    ,'GEN Higgs pT','Events','upplots/HpT'),
#        #"HAdR":     Hist(100,(0,2)      ,'GEN Higgs to A dR','Events','upplots/HAdR'),
#        #'HAdeta':   Hist(66 ,(-3.3,3.3) ,'GEN Higgs to A deta','Events','upplots/HAdeta'),
#        #'HAdphi':   Hist(66 ,(-3.3,3.3) ,'GEN Higgs to A dphi','Events','upplots/HAdphi'),
#        "A1pT":     Hist(80 ,(0,160)    ,'Highest GEN A pT','Events','upplots/A1pT'),
#        "A2pT":     Hist(80 ,(0,160)    ,'Lowest GEN A pT','Events','upplots/A2pT'),
#        "AdR":      Hist(50 ,(0,5)      ,'GEN A1 to A2 dR','Events','upplots/AdR'),
#        "bdRA1":    Hist(50 ,(0,5)      ,'GEN dR between highest pT A child bs','Events','upplots/bdRA1'),
#        "bdRA2":    Hist(50 ,(0,5)      ,'GEN dR between lowest pT A child bs','Events','upplots/bdRA2'),
#        "bdetaA1":  Hist(34 ,(0,3.4)    ,'GEN |deta| between highest-A child bs','Events','upplots/bdetaA1'),
#        "bdetaA2":  Hist(34 ,(0,3.4)    ,'GEN |deta| between lowest-A child bs','Events','upplots/bdetaA2'),
#        "bdphiA1":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between highest-A child bs','Events','upplots/bdphiA1'),
#        "bdphiA2":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between lowest-A child bs','Events','upplots/bdphiA2'),
#        "bphi":     Hist(66 ,(-3.3,3.3) ,'GEN b Phi','Events','upplots/bphi'),
#        "bjdR":     Hist(100,(0,2)      ,'All GEN bs to matched jet dR','Events','upplots/bjdR'),
#        "RjetpT":   Hist(100,(0,100)    ,'RECO matched jet pT','Events','upplots/RjetpT'),
#        "Rjeteta":  Hist(66 ,(-3.3,3.3) ,'RECO matched jet eta','Events','upplots/Rjeteta'),
#        "RjetCSVV2":Hist(140 ,(-12,2)    ,'RECO matched jet btagCSVV2 score','events','upplots/RjetCSVV2'),
#        "RjetDeepB":Hist(40 ,(-2.5,1.5) ,'RECO matched jet btagDeepB score','events','upplots/RjetDeepB'),
#        "RjetDeepFB"    :Hist(24 ,(0,1.2)    ,'RECO matched jet btagDeepFlavB score','events','upplots/RjetDeepFB'),
#        "RA1pT":    Hist(80 ,(0,160)    ,'pT of RECO A1 objects constructed from matched jets','Events','upplots/RA1pT'),
#        "RA2pT":    Hist(80 ,(0,160)    ,'pT of RECO A2 objects constructed from matched jets','Events','upplots/RA2pT'),
#        "RA1mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A1 objects from matched jets','Events','upplots/RA1mass'),
#        "RA2mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A2 objects from matched jets','Events','upplots/RA2mass'),
#        "RA1dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A1 object','Events','upplots/RA1dR'),
#        "RA2dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A2 object','Events','upplots/RA2dR'),
#        "RA1deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A1 object','Events','upplots/RA1deta'),
#        "RA2deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A2 object','Events','upplots/RA2deta'),
#        "RA1dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A1 object','Events','upplots/RA1dphi'),
#        "RA2dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A2 object','Events','upplots/RA2dphi'),
#        "RHmass":   Hist(80 ,(0,160)     ,'reconstructed mass of Higgs object from reconstructed As','Events','upplots/RHmass'),
#        "RHpT":     Hist(100,(0,200)    ,'pT of reconstructed higgs object from reconstructed As','Events','upplots/RHpT'),
#        "RHdR":     Hist(50 ,(0,5)      ,'dR between A children of reconstructed higgs object','Events','upplots/RHdR'),
#        "RHdeta":   Hist(33 ,(0,3.3)    ,'|deta| between A children of reconstructed higgs object','Events','upplots/RHdeta'),
#        "RHdphi":   Hist(33 ,(0,3.3)    ,'|dphi| between A children of reconstructed higgs object','Events','upplots/RHdphi'),
#        ##
#        "RalljetpT":    Hist(100,(0,100),'All RECO jet pT','Events','upplots/RalljetpT'),
#        "bjdRvlogbpT1":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 1st pT GEN b to matched RECO jet','upplots/bjdRvlogbpT1'),
#        "bjdRvlogbpT2":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 2nd pT GEN b to matched RECO jet','upplots/bjdRvlogbpT2'),
#        "bjdRvlogbpT3":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 3rd pT GEN b to matched RECO jet','upplots/bjdRvlogbpT3'),
#        "bjdRvlogbpT4":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 4th pT GEN b to matched RECO jet','upplots/bjdRvlogbpT4'),
#        "jetoverbpTvlogbpT1":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 1st GEN b pT for matched jets','upplots/jetoverbpTvlogbpT1'),
#        "jetoverbpTvlogbpT2":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 2nd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT2'),
#        "jetoverbpTvlogbpT3":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 3rd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT3'),
#        "jetoverbpTvlogbpT4":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 4th GEN b pT for matched jets','upplots/jetoverbpTvlogbpT4'),
#        "npassed":  Hist(1  ,(0.5,1.5) ,'','Number of events that passed cuts', 'upplots/npassed'),
#        "genAmass": Hist(40 ,(0,80)     ,'GEN mass of A objects','Events','upplots/Amass_g'),
#        "cutAmass": Hist(40 ,(0,80)     ,'GEN mass of A objects that pass cuts','Events','upplots/Amass_c')
    }
#    for plot in bjplots:
#        bjplots[plot].title = files[0]
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
        bgf = uproot.open(sigfiles[fnum])
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
            jets.cut(abs(jets.eta<2.4))
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
        
        print('Signal cut to ' + str(len(bs.eta)) + ' events')
        
        
        #############################
        # Constructing RECO objects #
        #############################



#        for prop in ['bpt','beta','bphi','bmass']:
#            jets[prop] = pd.DataFrame()
#            for i in range(nb):
#                jets[prop][i+1] = jets[prop[1:]][blist[i]>0].max(axis=1)
#                
#        jets.bdr = pd.DataFrame()
#        for i in range(nb):
#            jets.bdr[i+1] = blist[i][blist[i]>0].max(axis=1)
#            
#        ev.sync()
#            
#        if resjets==3:
#            pidx = [2,1,4,3]
#            for prop in ['bpt','beta','bphi','bmass']:
#                jets[prop]['merged'], jets[prop]['missing'] = (jets[prop][1]==jets[prop][3]),(jets[prop][1]==jets[prop][3])
#                for i in range(1,nb+1):
#                    jets[prop]['merged']=jets[prop]['merged']+jets[prop].fillna(0)[i][(jets.bmass[i]>=15) & (jets.bmass[i]+jets.bmass[pidx[i-1]]==jets.bmass[i])]
#                    jets[prop]['missing']=jets[prop]['missing']+jets[prop].fillna(0)[i][(jets.bmass[i]<15) & (jets.bmass[i]+jets.bmass[pidx[i-1]]==jets.bmass[i])]
#                    #jets[prop][i] = jets[prop][i]+(0*jets[prop][pidx])
#                    
#        bvec = []
#        for i in range(1,nb+1):
#            bvec.append(TLorentzVectorArray.from_ptetaphim(jets.bpt[i],jets.beta[i],jets.bphi[i],jets.bmass[i]))
#        
#        avec = []
#        for i in range(0,nb,2):
#            avec.append(bvec[i]+bvec[i+1])
#        
#        for prop in ['apt','aeta','aphi','amass']:
#            jets[prop] = pd.DataFrame()
#        for i in range(na):
#            jets.apt[i+1]  = avec[i].pt
#            jets.aeta[i+1] = avec[i].eta
#            jets.aphi[i+1] = avec[i].phi
#            jets.amass[i+1]= avec[i].mass
#        for prop in ['apt','aeta','aphi','amass']:
#            jets[prop].index = jets.pt.index
#        
#        hvec = [avec[0]+avec[1]]
#        
#        for prop in ['hpt','heta','hphi','hmass']:
#            jets[prop] = pd.DataFrame()
#        jets.hpt[1]  = hvec[0].pt
#        jets.heta[1] = hvec[0].eta
#        jets.hphi[1] = hvec[0].phi
#        jets.hmass[1]= hvec[0].mass
#        for prop in ['hpt','heta','hphi','hmass']:
#            jets[prop].index = jets.eta.index
#        
        #######################
        # Training Neural Net #
        #######################
        bgjetframe = pd.DataFrame()
        for i in range(njet):
            for prop in ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                bgjetframe['val'] = 0
        sigjetframe = pd.DataFrame()
        for i in range(njet):
            for prop in ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']:
                sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                sigjetframe['val'] = 1
        jetframe = pd.concat([bgjetframe,sigjetframe])

        X_train =jetframe.drop('val',axis=1).sample(frac=0.7, random_state=6)
        #Z_train=jetframe.sample(frac=0.7, random_state=6)
        X_test = jetframe.drop('val',axis=1).drop(X_train.index)
        #Z_test = jetframe.drop(X_train.index)
        Y_train =jetframe['val'].sample(frac=0.7, random_state=6)
        Y_test = jetframe['val'].drop(Y_train.index)
        
        #X_train, X_test, Y_train, Y_test = train_test_split(jetframe, jetframe['val'], test_size=0.3, random_state=0)



        model.fit(X_train, Y_train, epochs=50, batch_size=1)
        
        rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        print('Test accuracy:', test_acc)
        
        
        plots['DistStr'].fill(model.predict(X_train[Y_train==1]))
        plots['DistSte'].fill(model.predict(X_test[Y_test==1]))
        plots['DistBtr'].fill(model.predict(X_train[Y_train==0]))
        plots['DistBte'].fill(model.predict(X_test[Y_test==0]))
        plt.clf()
        plots['DistStr'].make(color='red',linestyle='-',htype='step')
        plots['DistBtr'].make(color='black',linestyle=':',htype='step')
        plots['DistSte'].make(color='red',linestyle='-',htype='step')
        plots['DistBte'].make(color='black',linestyle=':',htype='step')
        plots['Distribution'].plot(same=True)
        

        plt.clf()
        plt.plot([0,1],[0,1],'k--')
        plt.plot(rocx,rocy, label='Keras NN (area = {:.3f})'.format(auc(rocx,rocy)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('netplots/ROC')
    ############
    # Plotting #
    ############
        
#    plt.clf()
#    #plots.pop('bjdR').plot(logv=True)
#    for i in range(1,5):
#        bjplots.pop('s_bjdR'+str(i)).plot(logv=True)
#    for p in plots:
#        plots[p].plot()
#    for p in bjplots:
#        bjplots[p].plot()
#    #%%
#    if returnplots==True:
#        return plots
#    else:
#        sys.exit()

#def trig(sigfiles,bgfiles):
#    #%%
#    ## Create a dictionary of histogram objects
#    
#    #plots = {
#        #'cutflow':      Hist(4,(-0.5,4.5),'Total / Passed 4jet pT and |eta| cut/ passed DeepB > 0.4184','Events','recplots/datacutflow'),
#        #"RjetCSVV2":Hist([0,0.1241,0.4184,0.7527,1],None,'RECO matched jet btagCSVV2 score','events','recplots/dataCSVV2'),
#        #"RjetDeepB":Hist([0,0.0494,0.2770,0.7264,1],None,'RECO matched jet btagDeepB score','events','recplots/RjetDeepB'),
#        #"RjetDeepFB":Hist([0,0.0494,0.2770,0.7264,1],None,'RECO matched jet btagDeepFlavB score','events','recplots/dataDeepFB'),
#        
#    #}
#    #for plot in plots:
#        #plots[plot].title = files[0]
#    ## Create an internal figure for pyplot to write to
#    plt.figure(1)
#    ## Loop over all input files
#    nsig = len(sigfiles)
#    nbg = len(bgfiles)
#    smbg = nsig - nbg
#    for fnum in range(max(nsig, nbg)):
#        print('Opening ',sigfiles[fnum],' + ',bgfiles[fnum])
#        ## Open the file and retrieve our key branches
#        sigf = uproot.open(sigfiles[fnum])
#        bgf = uproot.open(bgfiles[fnum])
#        
#        sigevents = sigf.get('Events')
#        bgevents = bgf.get('Events')
#        
#        if smbg > 0:
#            print('Opening ',sigfiles[fnum+1],' + ',bgfiles[fnum],' to catch up')
#            sigf = uproot.open(sigfiles.pop(fnum+1))
#            sigevents
#        elif smbg < 0:
#            pass
#        
#        jets = PhysObj('jets')
#
#        jets.eta= pd.DataFrame(sigevents.array('Jet_eta')).rename(columns=inc)
#        jets.phi= pd.DataFrame(sigevents.array('Jet_phi')).rename(columns=inc)
#        jets.pt = pd.DataFrame(sigevents.array('Jet_pt')).rename(columns=inc)
#        jets.mass=pd.DataFrame(sigevents.array('Jet_mass')).rename(columns=inc)
#        jets.CSVV2 = pd.DataFrame(sigevents.array('Jet_btagCSVV2')).rename(columns=inc)
#        jets.DeepB = pd.DataFrame(sigevents.array('Jet_btagDeepB')).rename(columns=inc)
#        jets.DeepFB= pd.DataFrame(sigevents.array('Jet_btagDeepFlavB')).rename(columns=inc)
#
#        print('Processing ' + str(len(jets.eta)) + ' events')
#
#        ## Figure out how many bs and jets there are
#        njet= jets.eta.shape[1]
#
#        ## Fill 0 bin of cut flow plots
#        plots['cutflow'].fill(jets.pt.max(axis=1)*0)
#
#        ev = Event(jets)
#        jets.cut(abs(jets.eta)<2.4)
#        jets.cut(jets.pt>15)
#        resjets = (jets.pt/jets.pt).sum(axis=1)
#        resjets = resjets[resjets>=4]
#        plots['cutflow'].fill(resjets/resjets)
#        
#        plots['RjetCSVV2'].dfill(jets.CSVV2)
#        plots['RjetDeepFB'].dfill(jets.DeepFB)
#        
#        jets.cut(jets.DeepB > 0.4184 )
#        tagjets = (jets.DeepB/jets.DeepB).sum(axis=1)
#        tagjets = tagjets[tagjets>=2]
#        plots['cutflow'].fill(tagjets*2/tagjets)
#        
#        ev.sync()
#        
#
#    for pl in plots:
#        plots[pl].plot()
#        
#    print(plots['cutflow'][0],plots['cutflow'][1])
#
#
#    sys.exit()
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
                for i in sys.argv[idx+1:]:
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
                for i in sys.argv[idx+1:]:
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
 
def dialogue():
    print("Expected mndwrm.py <-f/-l> -s (signal.root) -b (background.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)

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