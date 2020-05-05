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
from tensorflow.python.keras import backend as BE



def main():
    #%%################
    # Plots and Setup #
    ###################

    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    netvars = ['pt','eta','phi','mass','CSVV2','DeepB','msoft','DDBvL']
    
    sigfile = 'GGHK1M_Nano.root'
    bgfile = 'TestData.root'
    datafiles = ['/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/003D724A-9341-2A40-A766-A663D3E4F10B.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/075AC8F4-7F0C-C447-82D1-CD6A47B26BCD.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/2CFFD279-3BCC-CF41-9577-B1A740DC2679.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/0F535BA4-C750-8E44-BD53-6B3011CA2AF8.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/039F2902-2B95-3B4A-9EC3-53EF6299F867.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/71369D3B-257F-524D-A55A-55968109677A.root',
                 '/cms/data/store/user/abrinke1/NanoAOD/2018/MC/QCD/QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/Nano25Oct2019/3542C35C-1109-D345-8B36-3DA027200467.root']
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
        "pt":       Hist(100,(0,200)    ,'pT for passing signal (red), BG (blue), and data (black) events','Events','netplots/pt'),
        "BGpt":     Hist(100,(0,200)),
        "SGpt":     Hist(100,(0,200)),
        "DTpt":     Hist(100,(0,200)),
        "eta":      Hist(100,(0,10)     ,'|eta| for passing signal (red), BG (blue), and data (black) events','Events','netplots/eta'),
        "BGeta":    Hist(100,(0,10)),
        "SGeta":    Hist(100,(0,10)),
        "DTeta":    Hist(100,(0,10)),
        "phi":      Hist(64 ,(-3.2,3.2) ,'phi for passing signal (red), BG (blue), and data (black) events','Events','netplots/phi'),
        "BGphi":    Hist(64 ,(-3.2,3.2)),
        "SGphi":    Hist(64 ,(-3.2,3.2)),
        "DTphi":    Hist(64 ,(-3.2,3.2)),
        "mass":     Hist(100,(0,100)    ,'mass for passing signal (red), BG (blue), and data (black) events','Events','netplots/mass'),
        "BGmass":   Hist(100,(0,100)),
        "SGmass":   Hist(100,(0,100)),
        "DTmass":   Hist(100,(0,100)),
        "CSVV2":    Hist(100,(-5,5)    ,'CSVV2 for passing signal (red), BG (blue), and data (black) events','Events','netplots/CSVV2'),
        "BGCSVV2":  Hist(100,(-5,5)),
        "SGCSVV2":  Hist(100,(-5,5)),
        "DTCSVV2":  Hist(100,(-5,5)),
        "DeepB":    Hist(100,(-5,5)    ,'DeepB for passing signal (red), BG (blue), and data (black) events','Events','netplots/DeepB'),
        "BGDeepB":  Hist(100,(-5,5)),
        "SGDeepB":  Hist(100,(-5,5)),
        "DTDeepB":  Hist(100,(-5,5)),
        "msoft":    Hist(100,(0,100)    ,'msoft for passing signal (red), BG (blue), and data (black) events','Events','netplots/msoft'),
        "BGmsoft":  Hist(100,(0,100)),
        "SGmsoft":  Hist(100,(0,100)),
        "DTmsoft":  Hist(100,(0,100)),
        "DDBvL":    Hist(100,(-5,5)    ,'DDBvL for passing signal (red), BG (blue), and data (black) events','Events','netplots/DDBvL'),
        "BGDDBvL":  Hist(100,(-5,5)),
        "SGDDBvL":  Hist(100,(-5,5)),
        "DTDDBvL":  Hist(100,(-5,5)),
    }
#    for plot in bjplots:
#        bjplots[plot].title = files[0]
#    for plot in plots:
#        plots[plot].title = files[0]

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    if True:
        
        #####################
        # Loading Variables #
        #####################
        print('Opening ',sigfile,', ',bgfile)
        
        ## Loop some data if the bg/signal files need to be equalized

            
        ## Open our file and grab the events tree
        sigf = uproot.open(sigfile)#'nobias.root')
        bgf = uproot.open(bgfile)
        
        sigevents = sigf.get('Events')
        bgevents = bgf.get('Events')
        dataevents = []
        for dfile in datafiles:
            print('Opening ',dfile)
            dataevents.append(uproot.open(dfile).get('Events'))

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
        
        datajets = [PhysObj('300'),PhysObj('500'),PhysObj('700'),PhysObj('1000'),PhysObj('1500'),PhysObj('2000'),PhysObj('inf')]
        for i in range(7):
            datajets[i].eta= pd.DataFrame(dataevents[i].array('FatJet_eta')).rename(columns=inc)
            datajets[i].phi= pd.DataFrame(dataevents[i].array('FatJet_phi')).rename(columns=inc)
            datajets[i].pt = pd.DataFrame(dataevents[i].array('FatJet_pt')).rename(columns=inc)
            datajets[i].mass=pd.DataFrame(dataevents[i].array('FatJet_mass')).rename(columns=inc)
            datajets[i].CSVV2 = pd.DataFrame(dataevents[i].array('FatJet_btagCSVV2')).rename(columns=inc)
            datajets[i].DeepB = pd.DataFrame(dataevents[i].array('FatJet_btagDeepB')).rename(columns=inc)
            datajets[i].DDBvL = pd.DataFrame(dataevents[i].array('FatJet_btagDDBvL')).rename(columns=inc)
            datajets[i].msoft = pd.DataFrame(dataevents[i].array('FatJet_msoftdrop')).rename(columns=inc)
            
        del dataevents
            
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
    
#        
        #######################
        # Training Neural Net #
        #######################
        bgjetframe = pd.DataFrame()
        for i in range(njet):
            for prop in netvars:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                #bgjetframe['val'] = 0
        sigjetframe = pd.DataFrame()
        for i in range(njet):
            for prop in netvars:
                sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                #sigjetframe['val'] = 1
        #X_train = pd.concat([bgjetframe.sample(frac=0.7,random_state=6),sigjetframe.sample(frac=0.7,random_state=6)])
        print('Signal cut to ',sigjetframe.shape[0], ' events')
        print('Background has ',bgjetframe.shape[0], ' events')
        
        dataweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]

        datapieces = []
        for weight in dataweights:
            tempframe = pd.DataFrame()
            for i in range(njet):
                for prop in netvars:
                    tempframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            tempframe = tempframe.sample(frac=weight,random_state=6)
            datapieces.append(tempframe)
            
        datajetframe = tempframe[0]
        for i in range(1,7):
            datajetframe = pd.concat(datajetframe,datapieces[i])
            
        del datapieces

        for col in netvars:
            plots['BG'+col].fill(bgjetframe[col])
            plots['SG'+col].fill(sigjetframe[col])
            plots['DT'+col].fill(datajetframe[col])
            
        plt.clf()
        for col in netvars:
            plots['BG'+col][0] = plots['BG'+col][0]/sum(plots['BG'+col][0])
            plots['SG'+col][0] = plots['SG'+col][0]/sum(plots['SG'+col][0])
            plots['DT'+col][0] = plots['DT'+col][0]/sum(plots['DT'+col][0])
            ##p.norm(sum(p[0]))
            #p[0] = p[0]/sum(p[0])
            
        for col in netvars:
            plt.clf()
            plots['SG'+col].make(color='red'  ,linestyle='-',htype='step')
            plots['BG'+col].make(color='blue' ,linestyle=':',htype='step')
            plots['DT'+col].make(color='black',linestyle=':',htype='step')
            plots[col].plot(same=True)
        
    #%%

    
if __name__ == "__main__":
    main()

#%%
