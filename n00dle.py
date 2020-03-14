#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility n00dle.py                               ###
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
#%%

def mc(files):
    ## This histogram object is used to accumulate and render our data, defined above
    pdgplt = Hist(40,(-0.5,39.5))
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open our file for processing
        f = uproot.open(files[fnum])
        events = f.get('Events')
        pdgida = events.array('GenPart_pdgId')
        parida = events.array('GenPart_genPartIdxMother')
        pdgid  = pd.DataFrame(pdgida)
        #parid  = pd.DataFrame(parida)
        #ppida  = pdgida[parida]
        #ppid   = pd.DataFrame(ppida)
        pt = events.array('GenPart_pt')
        print('Processing ' + str(len(pdgid)) + ' events')
        outlist = []
        #pdgid.replace([9000006,-9000006],36,inplace=True)
        #pdgplt.dfill(ppid[abs(pdgid)==5])
        # Loop over pdgid, using extremely slow ROOT-like logic instead of uproot logic.
        for event in range(pdgida.size):
            for iGen in range(pdgida[event].size):
                if abs(pdgida[event][iGen]) == 5:
                    parentIdx = parida[event][iGen]
                    if parentIdx == -1: continue
                    parentId = pdgida[event][parentIdx]
                    if abs(parentId) == 9000006:
                        outlist.append(36)
                        print(str(event) + " - " + str(iGen) + " = " + str(pt[event][iGen]))
#
                    else:
                        outlist.append(abs(parentId))
        # Fill out histogram with the list of values we obtained
        pdgplt.fill(outlist)
        
    
    plt.clf()  
    plot = pdgplt.make(logv=True) 
    plt.xlabel('Parent PdgId')
    plt.ylabel('Number of b Children')
    plt.savefig('upplots/parents.png')
    plt.show()
    print(plot)
    #return plot
    sys.exit()

def ana(files):
    #%%################
    # Plots and Setup #
    ###################
    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    #Aid = 36
    ## Make a dictionary of histogram objects
    bjplots = {}
    for i in range(1,5):
        bjplots.update({
        "s_beta"+str(i):      Hist(33 ,(-3.3,3.3)   ,'GEN b '+str(i)+' Eta (ranked by pT)','Events','upplots/s_beta'+str(i)),
        "s_bpT"+str(i):       Hist(60 ,(0,120)      ,'GEN pT of b '+str(i)+' (ranked by pT)','Events','upplots/s_bpT'+str(i)),
        "s_bjetpT"+str(i):    Hist(60 ,(0,120)      ,'Matched RECO jet '+str(i)+' pT (ranked by b pT)','Events','upplots/s_RjetpT'+str(i)),
        "s_bjeteta"+str(i):   Hist(33 ,(-3.3,3.3)   ,'Matched RECO jet '+str(i)+' Eta (ranked by b pT)','Events','upplots/s_Rjeteta'+str(i)),
        "s_bjdR"+str(i):      Hist(90 ,(0,3)        ,'GEN b '+str(i)+' (ranked by pT) to matched jet dR','Events','upplots/s_bjdR'+str(i))
        })
    plots = {
        "HpT":      Hist(60 ,(0,320)    ,'GEN Higgs pT','Events','upplots/HpT'),
        #"HAdR":     Hist(100,(0,2)      ,'GEN Higgs to A dR','Events','upplots/HAdR'),
        #'HAdeta':   Hist(66 ,(-3.3,3.3) ,'GEN Higgs to A deta','Events','upplots/HAdeta'),
        #'HAdphi':   Hist(66 ,(-3.3,3.3) ,'GEN Higgs to A dphi','Events','upplots/HAdphi'),
        "A1pT":     Hist(80 ,(0,160)    ,'Highest GEN A pT','Events','upplots/A1pT'),
        "A2pT":     Hist(80 ,(0,160)    ,'Lowest GEN A pT','Events','upplots/A2pT'),
        "AdR":      Hist(50 ,(0,5)      ,'GEN A1 to A2 dR','Events','upplots/AdR'),
        "bdRA1":    Hist(50 ,(0,5)      ,'GEN dR between highest pT A child bs','Events','upplots/bdRA1'),
        "bdRA2":    Hist(50 ,(0,5)      ,'GEN dR between lowest pT A child bs','Events','upplots/bdRA2'),
        "bdetaA1":  Hist(34 ,(0,3.4)    ,'GEN |deta| between highest-A child bs','Events','upplots/bdetaA1'),
        "bdetaA2":  Hist(34 ,(0,3.4)    ,'GEN |deta| between lowest-A child bs','Events','upplots/bdetaA2'),
        "bdphiA1":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between highest-A child bs','Events','upplots/bdphiA1'),
        "bdphiA2":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between lowest-A child bs','Events','upplots/bdphiA2'),
        "bphi":     Hist(66 ,(-3.3,3.3) ,'GEN b Phi','Events','upplots/bphi'),
        "bjdR":     Hist(100,(0,2)      ,'All GEN bs to matched jet dR','Events','upplots/bjdR'),
        "RjetpT":   Hist(100,(0,100)    ,'RECO matched jet pT','Events','upplots/RjetpT'),
        "Rjeteta":  Hist(66 ,(-3.3,3.3) ,'RECO matched jet eta','Events','upplots/Rjeteta'),
        "RjetCSVV2":Hist(140 ,(-12,2)    ,'RECO matched jet btagCSVV2 score','events','upplots/RjetCSVV2'),
        "RjetDeepB":Hist(40 ,(-2.5,1.5) ,'RECO matched jet btagDeepB score','events','upplots/RjetDeepB'),
        "RjetDeepFB"    :Hist(24 ,(0,1.2)    ,'RECO matched jet btagDeepFlavB score','events','upplots/RjetDeepFB'),
        "RA1pT":    Hist(80 ,(0,160)    ,'pT of RECO A1 objects constructed from matched jets','Events','upplots/RA1pT'),
        "RA2pT":    Hist(80 ,(0,160)    ,'pT of RECO A2 objects constructed from matched jets','Events','upplots/RA2pT'),
        "RA1mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A1 objects from matched jets','Events','upplots/RA1mass'),
        "RA2mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A2 objects from matched jets','Events','upplots/RA2mass'),
        "RA1dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A1 object','Events','upplots/RA1dR'),
        "RA2dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A2 object','Events','upplots/RA2dR'),
        "RA1deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A1 object','Events','upplots/RA1deta'),
        "RA2deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A2 object','Events','upplots/RA2deta'),
        "RA1dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A1 object','Events','upplots/RA1dphi'),
        "RA2dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A2 object','Events','upplots/RA2dphi'),
        "RHmass":   Hist(80 ,(0,160)     ,'reconstructed mass of Higgs object from reconstructed As','Events','upplots/RHmass'),
        "RHpT":     Hist(100,(0,200)    ,'pT of reconstructed higgs object from reconstructed As','Events','upplots/RHpT'),
        "RHdR":     Hist(50 ,(0,5)      ,'dR between A children of reconstructed higgs object','Events','upplots/RHdR'),
        "RHdeta":   Hist(33 ,(0,3.3)    ,'|deta| between A children of reconstructed higgs object','Events','upplots/RHdeta'),
        "RHdphi":   Hist(33 ,(0,3.3)    ,'|dphi| between A children of reconstructed higgs object','Events','upplots/RHdphi'),
        ##
        "RalljetpT":    Hist(100,(0,100),'All RECO jet pT','Events','upplots/RalljetpT'),
        "bjdRvlogbpT1":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 1st pT GEN b to matched RECO jet','upplots/bjdRvlogbpT1'),
        "bjdRvlogbpT2":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 2nd pT GEN b to matched RECO jet','upplots/bjdRvlogbpT2'),
        "bjdRvlogbpT3":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 3rd pT GEN b to matched RECO jet','upplots/bjdRvlogbpT3'),
        "bjdRvlogbpT4":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 4th pT GEN b to matched RECO jet','upplots/bjdRvlogbpT4'),
        "jetoverbpTvlogbpT1":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 1st GEN b pT for matched jets','upplots/jetoverbpTvlogbpT1'),
        "jetoverbpTvlogbpT2":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 2nd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT2'),
        "jetoverbpTvlogbpT3":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 3rd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT3'),
        "jetoverbpTvlogbpT4":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 4th GEN b pT for matched jets','upplots/jetoverbpTvlogbpT4'),
    }
    for plot in bjplots:
        bjplots[plot].title = files[0]
    for plot in plots:
        plots[plot].title = files[0]

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    
    ## Loop over input files
    for fnum in range(len(files)):
        
        #####################
        # Loading Variables #
        #####################
        
        print('Opening '+files[fnum])
        ## Open our file and grab the events tree
        f = uproot.open(files[fnum])#'nobias.root')
        events = f.get('Events')

        pdgida  = events.array('GenPart_pdgId')
        paridxa = events.array('GenPart_genPartIdxMother')
        parida  = pdgida[paridxa] 

        bs = PhysObj('bs')

        ## Removes all particles that do not have A parents 
        ## from the GenPart arrays, then removes all particles 
        ## that are not bs after resizing the pdgid array to be a valid mask
        
        bs.oeta = pd.DataFrame(events.array('GenPart_eta')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
        bs.ophi = pd.DataFrame(events.array('GenPart_phi')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
        bs.opt  = pd.DataFrame(events.array('GenPart_pt' )[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
        
        ## Test b order corresponds to As
        testbs = pd.DataFrame(events.array('GenPart_genPartIdxMother')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
        ## The first term checks b4 has greater idx than b1, the last two check that the bs are paired
        if ((testbs[4]-testbs[1]).min() <= 0) or ((abs(testbs[2]-testbs[1]) + abs(testbs[4])-testbs[3]).min() != 0):
            print('b to A ordering violated - time to do it the hard way')
            sys.exit()
        
        As = PhysObj('As')
        
        As.oeta = pd.DataFrame(events.array('GenPart_eta')[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        As.ophi = pd.DataFrame(events.array('GenPart_phi')[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        As.opt =  pd.DataFrame(events.array('GenPart_pt' )[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
        
        
        higgs = PhysObj('higgs')
        
        higgs.eta = pd.DataFrame(events.array('GenPart_eta')[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        higgs.phi = pd.DataFrame(events.array('GenPart_phi')[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        higgs.pt =  pd.DataFrame(events.array('GenPart_pt' )[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
        
        jets = PhysObj('jets')

        jets.eta= pd.DataFrame(events.array('Jet_eta')).rename(columns=inc)
        jets.phi= pd.DataFrame(events.array('Jet_phi')).rename(columns=inc)
        jets.pt = pd.DataFrame(events.array('Jet_pt')).rename(columns=inc)
        jets.mass=pd.DataFrame(events.array('Jet_mass')).rename(columns=inc)
        jets.CSVV2 = pd.DataFrame(events.array('Jet_btagCSVV2')).rename(columns=inc)
        jets.DeepB = pd.DataFrame(events.array('Jet_btagDeepB')).rename(columns=inc)
        jets.DeepFB= pd.DataFrame(events.array('Jet_btagDeepFlavB')).rename(columns=inc)

        print('Processing ' + str(len(bs.oeta)) + ' events')

        ## Figure out how many bs and jets there are
        nb = bs.oeta.shape[1]
        njet= jets.eta.shape[1]
        na = As.oeta.shape[1]
        if na != 2:
            print("More than two As per event, found "+str(na)+", halting")
            sys.exit()
            
        ## Create sorted versions of A values by pt
        for prop in ['eta','phi','pt']:
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
            
        ## Sort our b dataframes in descending order of pt
        for prop in ['spt','seta','sphi']:
            bs[prop] = pd.DataFrame()
        #bs.spt, bs.seta, bs.sphi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for i in range(1,nb+1):
                bs[prop][i] = bs[prop[1:]][bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            #bs.seta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            #bs.sphi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            

        ev = Event(bs,jets,As,higgs)
        jets.cut(jets.pt>0)
        bs.cut(bs.pt>0)
        ev.sync()
        
        ##############################
        # Processing and Calculation #
        ##############################

        ## Create our dR dataframe by populating its first column and naming it accordingly
        jbdr2 = pd.DataFrame(np.power(jets.eta[1]-bs.eta[1],2) + np.power(jets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
        sjbdr2= pd.DataFrame(np.power(jets.eta[1]-bs.seta[1],2) + np.power(jets.phi[1]-bs.sphi[1],2)).rename(columns={1:'Jet 1 b 1'})
        ## Loop over jet x b combinations
        jbstr = []
        for j in range(1,njet+1):
            for b in range(1,nb+1):
                ## Make our column name
                jbstr.append("Jet "+str(j)+" b "+str(b))
                if (j+b==2):
                    continue
                ## Compute and store the dr of the given b and jet for every event at once
                jbdr2[jbstr[-1]] = pd.DataFrame(np.power(jets.eta[j]-bs.eta[b],2) + np.power(jets.phi[j]-bs.phi[b],2))
                sjbdr2[jbstr[-1]]= pd.DataFrame(np.power(jets.eta[j]-bs.seta[b],2) + np.power(jets.phi[j]-bs.sphi[b],2))
        
        ## Create a copy array to collapse in jets instead of bs
        blist = []
        sblist = []
        for b in range(nb):
            blist.append(jbdr2.filter(like='b '+str(b+1)))
            blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
            blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))
            sblist.append(sjbdr2.filter(like='b '+str(b+1)))
            sblist[b] = sblist[b][sblist[b].rank(axis=1,method='first') == 1]
            sblist[b] = sblist[b].rename(columns=lambda x:int(x[4:6]))
        
        ## Cut our events to only resolved 4jet events with dR<0.4
        rjets = blist[0][blist[0]<0.4].fillna(0)
        for i in range(1,4):
            rjets = np.logical_or(rjets,blist[i][blist[i]<0.4].fillna(0))
        rjets = rjets.sum(axis=1)
        rjets = rjets[rjets==4].dropna()
        jets.trimTo(rjets)
        ev.sync()
        
        #############################
        # Constructing RECO objects #
        #############################

        for prop in ['bpt','beta','bphi','bmass']:
            jets[prop] = pd.DataFrame()
            for i in range(nb):
                jets[prop][i+1] = jets[prop[1:]][blist[i]>0].max(axis=1)
                
        jets.bdr = pd.DataFrame()
        for i in range(nb):
            jets.bdr[i+1] = np.sqrt(blist[i][blist[i]>0].max(axis=1))
            
        ev.sync()
            
        bvec = []
        for i in range(1,nb+1):
            bvec.append(TLorentzVectorArray.from_ptetaphim(jets.bpt[i],jets.beta[i],jets.bphi[i],jets.bmass[i]))
        
        avec = []
        for i in range(0,nb,2):
            avec.append(bvec[i]+bvec[i+1])
        
        for prop in ['apt','aeta','aphi','amass']:
            jets[prop] = pd.DataFrame()
        for i in range(na):
            jets.apt[i+1]  = avec[i].pt
            jets.aeta[i+1] = avec[i].eta
            jets.aphi[i+1] = avec[i].phi
            jets.amass[i+1]= avec[i].mass
        for prop in ['apt','aeta','aphi','amass']:
            jets[prop].index = jets.pt.index
        
        hvec = [avec[0]+avec[1]]
        
        for prop in ['hpt','heta','hphi','hmass']:
            jets[prop] = pd.DataFrame()
        jets.hpt[1]  = hvec[0].pt
        jets.heta[1] = hvec[0].eta
        jets.hphi[1] = hvec[0].phi
        jets.hmass[1]= hvec[0].mass
        for prop in ['hpt','heta','hphi','hmass']:
            jets[prop].index = jets.eta.index
        
        ################
        # Filling Data #
        ################
        
        for i in range(4):
            plots['bjdRvlogbpT'+str(i+1)].dfill(np.log2(bs.spt[[i+1]]),bs.trim(sblist[i]))
            plots['RjetCSVV2'].dfill(jets.CSVV2[blist[i]>0])
            plots['RjetDeepB'].dfill(jets.DeepB[blist[i]>0])
            plots['RjetDeepFB'].dfill(jets.DeepFB[blist[i]>0])

            yval = np.divide(jets.pt[sblist[i]>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0],bs.spt[[i+1]].dropna().reset_index(drop=True)[i+1])
            xval = np.log2(bs.spt[[i+1]]).melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0]
            plots['jetoverbpTvlogbpT'+str(i+1)].fill(xval,yval)

            bjplots['s_bpT'+str(i+1)].dfill(bs.spt[[i+1]])
            bjplots['s_beta'+str(i+1)].dfill(bs.seta[[i+1]])
            bjplots['s_bjetpT'+str(i+1)].dfill(jets.pt[sblist[i]>0])
            bjplots['s_bjeteta'+str(i+1)].dfill(jets.eta[sblist[i]>0])
            bjplots['s_bjdR'+str(i+1)].dfill(np.sqrt(sblist[i][sblist[i]!=0]))

        plots['HpT'].dfill(higgs.pt)
        plots['A1pT'].fill(As.pt[1])
        plots['A2pT'].fill(As.pt[2])
        plots['AdR'].fill(np.sqrt(np.power(As.eta[2]-As.eta[1],2) + np.power(As.phi[2]-As.phi[1],2)))
        plots['bdRA1'].fill(np.sqrt(np.power(bs.eta[2]-bs.eta[1],2) + np.power(bs.phi[2]-bs.phi[1],2)))
        plots['bdRA2'].fill(np.sqrt(np.power(bs.eta[4]-bs.eta[3],2) + np.power(bs.phi[4]-bs.phi[3],2)))
        plots['bdetaA1'].fill(abs(bs.eta[2]-bs.eta[1]))
        plots['bdetaA2'].fill(abs(bs.eta[4]-bs.eta[3]))
        plots['bdphiA1'].fill(abs(bs.phi[2]-bs.phi[1]))
        plots['bdphiA2'].fill(abs(bs.phi[4]-bs.phi[3]))
        

        
        plots['bphi'].dfill(bs.phi)
        plots['RalljetpT'].dfill(jets.pt)
        plots['bjdR'].dfill(jets.bdr)
        plots['RjetpT'].dfill(jets.bpt)
        plots['Rjeteta'].dfill(jets.beta)  
        for i in range(1,3):
            plots['RA'+str(i)+'pT'  ].fill(jets.apt[i])
            plots['RA'+str(i)+'mass'].fill(jets.amass[i])
            plots['RA'+str(i)+'deta'].fill(abs(jets.beta[2*i]-jets.beta[(2*i)-1]))
            plots['RA'+str(i)+'dR'  ].fill(np.sqrt(np.power(jets.beta[2*i]-jets.beta[(2*i)-1],2)+np.power(jets.bphi[2*i]-jets.bphi[(2*i)-1],2)))
            plots['RA'+str(i)+'deta'].fill(abs(jets.beta[2*i]-jets.beta[(2*i)-1]))
            plots['RA'+str(i)+'dphi'].fill(abs(jets.bphi[2*i]-jets.bphi[(2*i)-1]))
        plots['RHpT'  ].fill(jets.hpt[1])
        plots['RHmass'].fill(jets.hmass[1])
        plots['RHdR'  ].fill(np.sqrt(np.power(jets.aeta[2]-jets.aeta[1],2)+np.power(jets.aphi[2]-jets.aphi[1],2)))
        plots['RHdeta'].fill(abs(jets.aeta[2]-jets.aeta[1]))
        plots['RHdphi'].fill(abs(jets.aphi[2]-jets.aphi[1]))
    
    ############
    # Plotting #
    ############
        
    plt.clf()
    plots.pop('bjdR').plot(logv=True)
    for i in range(1,5):
        bjplots.pop('s_bjdR'+str(i)).plot(logv=True)
    for p in plots:
        plt.clf()
        plots[p].plot()
    for p in bjplots:
        plt.clf()
        bjplots[p].plot()
    #%%
    sys.exit()


def trig(files):
    ## Create a dictionary of histogram objects
    rptbins = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,30,100]
    plots = {
        'hptplot':      Hist(rptbins,None,'Highest Muon pT','Events passing HLT','upplots/TrigHpTplot'),
        'ptplot':       Hist(rptbins,None,'Highest Muon pT','Events','upplots/TrigpTplot'),
        'ratioptplot':  Hist(rptbins,None,'Highest Muon pT','HLT_Mu7_IP4 / Events with Muons of sip > 5','upplots/TrigRatiopTPlot'),
        'sipplot':      Hist(20,(0,20),'Highest Muon SIP', 'Events', 'upplots/TrigSIPplot'),
        'hsipplot':     Hist(20,(0,20),'Highest Muon SIP', 'Events', 'upplots/TrigHSIPplot'),
        'ratiosipplot': Hist(20,(0,20),'Highest Muon SIP', 'HLT_Mu7_IP4 / Events with muons of pT > 10', 'upplots/TrigRatioSIPplot'),
        'HLTcutflow':      Hist(12,(-0.5,11.5),'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6','Events','upplots/cutflowHLT'),
        'L1Tcutflow':      Hist(12,(-0.5,11.5),'All // L1_SingleMu6/7/8/9/10/12/14/16/18','Events','upplots/cutflowL1T'),
        'HLTcutflowL':      Hist(12,(-0.5,11.5),'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6','Events','upplots/cutflowHLT-L'),
        'L1TcutflowL':      Hist(12,(-0.5,11.5),'All // L1_SingleMu6/7/8/9/10/12/14/16/18','Events','upplots/cutflowL1T-L')

    }
    cutflow2d = Hist2d([9,10],[[-0.5,8.5],[-0.5,9.5]],'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6',
        'All // L1_SingleMu6/7/8/9/10/12/14/16/18','upplots/cutflowHLTvsL1T',files[0])
    for plot in plots:
        plots[plot].title = files[0]
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over all input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open the file and retrieve our key branches
        f = uproot.open(files[fnum])
        events = f.get('Events')

        HLTcuts = ['HLT_Mu7_IP4','HLT_Mu8_IP3','HLT_Mu8_IP5','HLT_Mu8_IP6','HLT_Mu9_IP4','HLT_Mu9_IP5','HLT_Mu9_IP6','HLT_Mu12_IP6']
        L1Tcuts = ['L1_SingleMu6','L1_SingleMu7','L1_SingleMu8','L1_SingleMu9','L1_SingleMu10','L1_SingleMu12','L1_SingleMu14','L1_SingleMu16','L1_SingleMu18']

        Muon = PhysObj('Muon',files[fnum],'pt','eta','phi','sip3d','mediumId')
        Trig = PhysObj('trig')
        HLT = PhysObj('HLTrig')
        L1T = PhysObj('L1Trig')
        Trig.vals = pd.DataFrame(events.array('HLT_Mu7_IP4_part0')).rename(columns=inc)
        for tr in HLTcuts:
            HLT[tr] = pd.DataFrame(events.array(tr+'_part0')).rename(columns=inc)
        for tr in L1Tcuts:
            L1T[tr]= pd.DataFrame(events.array(tr+'er1p5')).rename(columns=inc)
        ev = Event(Muon,Trig,HLT,L1T)
        print('Processing ' + str(len(Muon.pt)) + ' events')
   
        ## Fill 0 bin of cut flow plots

        plots['HLTcutflow'].dfill(HLT[HLTcuts[0]]*0)
        plots['L1Tcutflow'].dfill(L1T[L1Tcuts[0]]*0)
        cutflow2d.dfill(HLT[HLTcuts[0]]*0,HLT[HLTcuts[0]]*0)

 
        ## Fill the rest of the bins
        ct = 1
        for i in HLT:
            plots['HLTcutflow'].dfill(HLT[i][HLT[i]].dropna()*ct)
            cutflow2d.dfill(HLT[i][HLT[i]].dropna()*ct,HLT[i][HLT[i]].dropna()*0)
            ct = ct + 1
        ct = 1
        for i in L1T:
            plots['L1Tcutflow'].dfill(L1T[i][L1T[i]].dropna()*ct)
            cutflow2d.dfill(L1T[i][L1T[i]].dropna()*0,L1T[i][L1T[i]].dropna()*ct)
            ct = ct + 1

        ht = 1
        for i in HLT:
            lt = 1
            for j in L1T:
                cutflow2d.dfill(HLT[i][HLT[i] & L1T[j]].dropna()*ht,L1T[j][L1T[j] & HLT[i]].dropna()*lt)
                lt = lt + 1
            ht = ht + 1

        ##Perform global cuts
        Muon.cut(abs(Muon.eta)<1.5)
        Muon.cut(Muon.mediumId==True)
        ev.sync()

        ##Fill bin 1 of cut flow lots

        #plots['HLTcutflow'].fill((Muon.pt/Muon.pt).max(axis=1).dropna())
        #plots['L1Tcutflow'].fill((Muon.pt/Muon.pt).max(axis=1).dropna())


        ## Cut muons and trim triggers to the new size
        MuonP = Muon.cut(Muon.sip3d>5,split=True)
        MuonS = Muon.cut(Muon.pt>10,split=True)
        TrigP = Trig.trimTo(MuonP.pt,split=True)
        TrigS = Trig.trimTo(MuonS.sip3d,split=True)
        ## Reshape triggers to fit our muons
        for i in MuonP.pt.columns:
            TrigP.vals[i] = TrigP.vals[1]
        for i in MuonS.sip3d.columns:
            TrigS.vals[i] = TrigS.vals[1]

        ## Create the two histograms we want to divide
        plt.figure(1)
        plots['ptplot'].fill(MuonP.pt.max(axis=1))
        plots['hptplot'].fill(MuonP.pt[TrigP.vals].max(axis=1).dropna(how='all'))
        plots['sipplot'].fill(MuonS.sip3d.max(axis=1))
        plots['hsipplot'].fill(MuonS.sip3d[TrigS.vals].max(axis=1).dropna(how='all'))
    plots['ratioptplot'].add(plots['hptplot'].divideby(plots['ptplot'],split=True))
    plots['ratiosipplot'].add(plots['hsipplot'].divideby(plots['sipplot'],split=True))
    plots['HLTcutflowL'].add(plots['HLTcutflow'])
    plots['L1TcutflowL'].add(plots['L1Tcutflow'])
    cutflow2d.norm()[0][0][0] = 0
    cutflow2d.plot(text=True,edgecolor='black')
    plots.pop('HLTcutflowL').norm().plot(ylim=(None,.2))
    plots.pop('L1TcutflowL').norm().plot(ylim=(None,.2))
    plots.pop('HLTcutflow').norm().plot()
    plots.pop('L1Tcutflow').norm().plot()

    for pl in plots:
        plots[pl].plot()
    sys.exit()

def main():
    if (len(sys.argv) > 1): 
        files=[]
        ## Check for file sources
        if '-f' in sys.argv:
            idx = sys.argv.index('-f')+1
            for i in sys.argv[idx:]:
                files.append(i)
        elif '-l' in sys.argv:
            with open(sys.argv[3],'r') as rfile:
                for line in rfile:
                    files.append(line.strip('\n')) 
        else:
            files.append('NMSSM-20.root')
        ## Check specified run mode
        if sys.argv[1] == '-mc':
            mc(files)
        elif sys.argv[1] == '-trig':
            trig(files)
        elif sys.argv[1] == '-a':
            ana(files)
 
    print("Expected n00dle.py <switch> (flag) (target)")
    print("-----switches-----")
    print("-mc    Runs a b-parent analysis on MC")
    print("-trig  Analyzes trigger efficiency for data")
    print("-a     Analyzes jet-b correlations")
    print("---optional flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)
    

## Define 'main' function as primary executable
if __name__ == '__main__':
    main()
