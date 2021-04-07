#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:42:03 2021

"""
import numpy as np
from analib import inc, Hist, PhysObj, Event, dphi
import pandas as pd
import copy as cp
import uproot, json, sys, pickle
import matplotlib.pyplot as plt

def lumipucalc(inframe):
    for var in ['extweight','mpt','meta','mip','npvsG']:
        if var not in inframe.columns:
            raise ValueError(f"Dataframe passed to lumipucalc() with no {var} column")
    Rtensor = pickle.load(open('MuonRtensor.p',"rb"))
    Ltensor = pickle.load(open('MuonLtensor.p',"rb"))
    Rmeta = Rtensor.pop('meta')
    Lmeta = Ltensor.pop('meta')
    
    pd.set_option('mode.chained_assignment',None)
    
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
                inframe['extweight'][(inframe['mpt'] >= x)&(inframe['mpt'] < xn)&\
                       (inframe['mip'] >= y)&(inframe['mip'] < yn)&(abs(inframe['meta']) < 1.5)&\
                       (inframe['npvsG'] == b+1)] = inframe['extweight'] * Rtensor[x][y]['L'][b] * Ltensor[x][y]['L']
            
                inframe['extweight'][(inframe['mpt'] >= x)&(inframe['mpt'] < xn)&\
                        (inframe['mip'] >= y)&(inframe['mip'] < yn)&(abs(inframe['meta']) >= 1.5)&\
                        (inframe['npvsG'] == b+1)] = inframe['extweight'] * Rtensor[x][y]['H'][b] * Ltensor[x][y]['H']
            
            inframe['extweight'][(inframe['mpt'] >= x)&(inframe['mpt'] < xn)&\
                       (inframe['mip'] >= y)&(inframe['mip'] < yn)&(abs(inframe['meta']) < 1.5)&\
                       (inframe['npvsG'] > b+1)] = inframe['extweight'] * Rtensor[x][y]['L'][b] * Ltensor[x][y]['L']
            
            inframe['extweight'][(inframe['mpt'] >= x)&(inframe['mpt'] < xn)&\
                (inframe['mip'] >= y)&(inframe['mip'] < yn)&(abs(inframe['meta']) >= 1.5)&\
                (inframe['npvsG'] > b+1)] = inframe['extweight'] * Rtensor[x][y]['H'][b] * Ltensor[x][y]['H']
                    
    return inframe['extweight']

def loadjets(jets, events,bgweights=False):
    jets.eta= pd.DataFrame(events.array('FatJet_eta')).rename(columns=inc)
    jets.phi= pd.DataFrame(events.array('FatJet_phi')).rename(columns=inc)
    jets.pt = pd.DataFrame(events.array('FatJet_pt' )).rename(columns=inc)
    jets.mass=pd.DataFrame(events.array('FatJet_mass')).rename(columns=inc)
    jets.CSVV2 = pd.DataFrame(events.array('FatJet_btagCSVV2')).rename(columns=inc)
    jets.DeepB = pd.DataFrame(events.array('FatJet_btagDeepB')).rename(columns=inc)
    jets.DDBvL = pd.DataFrame(events.array('FatJet_btagDDBvL')).rename(columns=inc)
    jets.msoft = pd.DataFrame(events.array('FatJet_msoftdrop')).rename(columns=inc)
    jets.H4qvs = pd.DataFrame(events.array('FatJet_deepTagMD_H4qvsQCD')).rename(columns=inc)
    jets.n2b1  = pd.DataFrame(events.array('FatJet_n2b1')).rename(columns=inc)
    jets.n2b1[jets.n2b1 < -5] = -2

    jets.event = pd.DataFrame(events.array('event')).rename(columns=inc)
    jets.npvs  = pd.DataFrame(events.array('PV_npvs')).rename(columns=inc)
    jets.npvsG = pd.DataFrame(events.array('PV_npvsGood')).rename(columns=inc)
    
    # idxa1 = events.array('FatJet_subJetIdx1')
    # idxa2 = events.array('FatJet_subJetIdx2')
    # idxa1f = pd.DataFrame(idxa1).rename(columns=inc)
    # idxa2f = pd.DataFrame(idxa2).rename(columns=inc)
    # submass = events.array('SubJet_mass')
    # subtau = events.array('SubJet_tau1')
    # jets.submass1 = pd.DataFrame(submass[idxa1[idxa1!=-1]]).rename(columns=inc).add(idxa1f[idxa1f==-1]*0,fill_value=0)
    # jets.submass2 = pd.DataFrame(submass[idxa2[idxa2!=-1]]).rename(columns=inc).add(idxa2f[idxa2f==-1]*0,fill_value=0)
    # jets.subtau1  = pd.DataFrame(subtau[ idxa1[idxa1!=-1]]).rename(columns=inc).add(idxa1f[idxa1f==-1]*0,fill_value=0)
    # jets.subtau2  = pd.DataFrame(subtau[ idxa2[idxa2!=-1]]).rename(columns=inc).add(idxa2f[idxa2f==-1]*0,fill_value=0)
    # del idxa1, idxa2, idxa1f, idxa2f, submass, subtau
    
    jets.extweight = jets.event / jets.event
    if bgweights:
        jets.HT = pd.DataFrame(events.array('LHE_HT')).rename(columns=inc)
        tempweight = 4.346 - (0.356*np.log2(jets.HT[1]))
        tempweight[tempweight < 0.1] = 0.1
        jets.extweight[1] = jets.extweight[1] * tempweight
        
    else:
        jets.HT = jets.event * 6000 / jets.event

    for j in range(1,jets.pt.shape[1]):
        jets.event[j+1] = jets.event[1]
        jets.npvs[j+1] = jets.npvs[1]
        jets.npvsG[j+1] = jets.npvsG[1]
        jets.extweight[j+1] = jets.extweight[1]
        jets.HT[j+1] = jets.HT[1]
    return jets

def computedR(jet,thing,nms=['jet','thing']):
    nj = jet.eta.shape[1]
    nt = thing.eta.shape[1]
    ## Create our dR dataframe by populating its first column and naming it accordingly
    jtdr2 = pd.DataFrame(np.power(jet.eta[1] - thing.eta[1],2) + np.power(dphi(jet.phi[1],thing.phi[1]),2)).rename(columns={1:f"{nms[0]} 1 {nms[1]} 1"})
    jtstr = []
    ## Loop over jet x thing combinations
    for j in range(1,nj+1):
        for t in range(1,nt+1):
            jtstr.append(f"{nms[0]} {j} {nms[1]} {t}")
            if (j+t==2):
                continue
            jtdr2[jtstr[-1]] = pd.DataFrame(np.power(jet.eta[j]-thing.eta[t],2) + np.power(dphi(jet.phi[j],thing.phi[t]),2))
    return np.sqrt(jtdr2)

#%%
def compare(conf,option,stage):
#%%    
    print(f"Analysing {conf} with {option} cuts")
    with open(conf) as f:
        confd = json.load(f)
        islhe =     confd['islhe']
        isdata =    confd['isdata']
        files =     confd['files']
        if type(files) != list:
            files = [files]
        fweights =  confd['weight']
        if type(fweights) != list:
            fweights = [fweights]
        name =      confd['name']
        
        
    if stage == "A":
        numplot = {'pt':    Hist(27,(150,1500)  ,'pT of AK8 jet passing cuts + triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_pTEfficiencyPlot_{option}_A"),
               }
    elif stage == "B":
        numplot = {'pt':    Hist(50,(0,1000)    ,'pT of AK8 jet passing cuts + triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_pTEfficiencyPlot_{option}_B"),
                   'msoft': Hist(11,(90,200)    ,'softdrop mass of AK8 jet above 400GeV passing cuts+triggers / cuts',f"{option} Ratio",f"Effplots/{name}_msoftEfficiencyPlot_{option}_B"),
                   'DDBvL': Hist(20,(0.8,1.0)   ,'DDBvL of AK8 jet above 400GeV passing cuts+triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_ddbvlEfficiencyPlot_{option}_B"),
                   }
    elif stage == "C":
        numplot = {'pt':    Hist(27,(150,1500)  ,'pT of AK8 jet passing cuts + triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_pTEfficiencyPlot_{option}_C"),
                   's2pt':  Hist(61,(30,1050)   ,'pT of 2nd highest pT slimjet passing cuts+triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_s2pTEfficiencyPlot_{option}_C"),
                   'lowb':  Hist(20,(0,1.0)     ,'Lowest deepB of two slimjets passing cuts+triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_lowbEfficiencyPlot_{option}_C"),
               }
    elif stage == "D":
        numplot = {'pt':    Hist(27,(150,1500)  ,'pT of AK8 jet passing cuts + triggers / passing cuts',f"{option} Ratio",f"Effplots/{name}_pTEfficiencyPlot_{option}_D"),
                   }
    elif len(stage)>1:
        numplot = {'pt':    Hist(5,(400,650)    ,'pT of AK8 jet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_ptScalePlot"),
                   'msoft': Hist(6,(80,200)     ,'msoft of AK8 jet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_{option}_msoftScalePlot"),
                   'DDBvL': Hist(4,(.80,1)      ,'DDBvL of AK8 jet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_{option}_ddbvlScalePlot"),
                   }
        # numplot['pt'][1][-1] = np.inf
        # numplot['msoft'][1][0] = 90
        
        if "C" in stage:
            numplot.update({
                    's2pt':     Hist(4,(140,220)                ,'pT of 2nd highest pT slimjet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_{option}_s2ptScalePlot"),
                    's2deepb':  Hist([.4184,.5856,.7527,0.8764,1],None ,'DeepB of 2nd highest DeepB slimjet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_{option}_s2deepbScalePlot"),
                })
        if stage == "CX":
            numplot.update({'pt':       Hist(3,(250,400)    ,'pT of AK8 jet passing cuts+triggers / cuts',f"{option} Ratio",f"Seffplots/{name}_{stage}_ptScalePlot"),
                            })
        
        # for p in numplot:
        #     numplot[p][1][0] = 0
        
        

        
        
    for p in numplot:
        numplot[p].title  = f"{name} {stage}"
        numplot[p].ylim   = (0,1)
        
    denplot = cp.deepcopy(numplot)
    
    
    elecvars = ['pt','eta','mvaFall17V2Iso_WP90']
    muvars = ['pt','eta','mediumPromptId','miniIsoId','softId','dxy','dxyErr','ip3d']
    l1vars = ['SingleJet180','Mu7_EG23er2p5','Mu7_LooseIsoEG20er2p5','Mu20_EG10er2p5','SingleMu22',
              'SingleMu25','DoubleJet112er2p3_dEta_Max1p6','DoubleJet150er2p5']
    hltvars = ['AK8PFJet500','Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ','Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
               'Mu27_Ele37_CaloIdL_MW','Mu37_Ele27_CaloIdL_MW',
               'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
               'DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71']
    slimvars = ['pt','eta','phi','btagDeepB','puId']
    
    print("Collecting event information")
    
    events, jets, elecs, mus, l1s, hlts, sjets = [],[],[],[],[],[],[]
    for i in range(len(files)):
        events.append(uproot.open(files[i]).get('Events'))
        
        jets.append(loadjets(PhysObj(f"Jets{i}"),events[i], islhe))
        jets[i].extweight = jets[i].extweight * fweights[i]
        
        
        elecs.append(PhysObj(f"Electron{i}", files[i], *elecvars, varname='Electron'))
        
        mus.append(PhysObj(f"Muon{i}", files[i], *muvars, varname='Muon'))
        mus[i].eta = abs(mus[i].eta)
        mus[i].ip = abs(mus[i].dxy / mus[i].dxyErr)
        
        l1s.append(PhysObj(f"L1{i}", files[i], *l1vars, varname='L1'))
        hlts.append(PhysObj(f"HLT{i}",files[i],*hltvars,varname='HLT'))
        
        sjets.append(PhysObj(f"Slimjet{i}",files[i],*slimvars,varname='Jet'))
        
        
    evs = []
    for i in range(len(files)):
        evs.append(Event(jets[i],l1s[i],hlts[i],elecs[i],mus[i]))
        if "C" in stage:
            evs[i].register(sjets[i])
            
        
    for jet in jets:
        jet.cut(jet.pt > 170)
        jet.cut(abs(jet.eta)<2.4)
        jet.cut(jet.DDBvL > 0.8)
        jet.cut(jet.DeepB > 0.4184)
        jet.cut(jet.msoft > 90)
        jet.cut(jet.mass > 90)
        jet.cut(jet.msoft < 200)
        jet.cut(jet.npvsG >= 1)
        if "AB" in stage:
            jet.cut(jet.pt >= 400)
        elif stage == "CX":
            jet.cut(jet.pt >= 250)
            jet.cut(jet.pt < 400)
            
        
        
    if option == 'MuonEG':
        for elec in elecs:
            elec.cut(elec.pt > 15)
            elec.cut(abs(elec.eta) < 2.5)
            elec.cut(elec.mvaFall17V2Iso_WP90 > 0.9) 
        
    for mu in mus:
        if option == 'MuonEG':
            mu.cut(mu.pt > 10)
            mu.cut(abs(mu.eta) < 2.4)
            mu.cut(mu.mediumPromptId > 0.9)
            mu.cut(mu.miniIsoId >= 2)
        elif option == 'Parked':
            mu.cut(mu.softId > 0.9)
            mu.cut(abs(mu.eta) < 2.4)
            mu.cut(mu.pt > 7)
            mu.cut(mu.ip > 2)
            mu.cut(mu.ip3d < 0.5)
        #else: raise(NameError("Dataset name does not match expected"))
        
    for ev in evs: ev.sync()
    
    
    
    if option == 'MuonEG':
        for i in range(len(files)):
            l1s[i].cut(np.logical_or.reduce((
                np.logical_and(np.logical_or(l1s[i].Mu7_EG23er2p5,l1s[i].Mu7_LooseIsoEG20er2p5),hlts[i].Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ),
                np.logical_and(np.logical_or(l1s[i].Mu20_EG10er2p5,l1s[i].SingleMu22),hlts[i].Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL),
                np.logical_and(l1s[i].SingleMu25,hlts[i].Mu27_Ele37_CaloIdL_MW),
                np.logical_and(l1s[i].SingleMu25,hlts[i].Mu37_Ele27_CaloIdL_MW))))
    
            ## Makes a frame whose elements have the highest pt of the muon or electron in that position
            passf = elecs[i].pt.combine(mus[i].pt,np.maximum,fill_value=0)
            ## Drops pt < 25
            passf = passf[passf > 25]
            ## Drops empty rows
            passf = passf.dropna(how='all')
            ## The remaining events must have had an electron or muon with pt > 25 - the rest are removed
            elecs[i].trimTo(passf)
        
    for ev in evs: ev.sync()
    
    if "C" in stage or stage == "AB":
        for i in range(len(files)):
            print(f"Processing file {i} slimjets")
            jets[i].cut(jets[i].pt.rank(axis=1,method='first',ascending=False) == 1)              

            sjets[i].cut(abs(sjets[i].eta) < 2.4)
            sjets[i].cut(sjets[i].pt > 30)
            sjets[i].cut(sjets[i].puId >= 1)
            sjets[i].cut(sjets[i].pt > 140)
            sjets[i].cut(sjets[i].btagDeepB > 0.4184)
            evs[i].sync()
            ## This entire block is designed to remove any events whose defined a and b jets
            ## have a dR > 0.8 to the highest pT passing jet
            print("Computing dR")
            if sjets[i].pt.shape[0] < 1 or jets[i].pt.shape[0] < 1:
                continue
            sjjdr = computedR(jets[i],sjets[i],['Fatjet','slimjet'])
            jlist = []
            print("Assembling slim frame")
            for j in range(jets[i].pt.shape[1]):
                jlist.append(sjjdr.filter(like=f"Fatjet {j+1}"))
                jlist[j] = jlist[j].rename(columns=lambda x:int(x[-2:]))
            jlist[0][jlist[0] == 0] = jlist[0]+0.001
            sjframe = jlist[0][jlist[0] < 0.8].fillna(0)
            for j in range(1,jets[i].pt.shape[1]):
                jlist[j][jlist[j] == 0] = jlist[j]+0.001
                sjframe = sjframe + jlist[j][jlist[j] < 0.8].fillna(0)
            sjets[i].cut(sjframe!=0)
            ## Trims the collection of slimjets < dR 0.8 to only events with a 2nd passing jet
            sjets[i].trimto(sjets[i].cut(sjets[i].pt.rank(axis=1,method='first',ascending=False) == 2,split=True).pt)
            if stage == "AB":
                for elem in sjets[i]:
                    evs[i].frame = evs[i].frame.loc[evs[i].frame.index.difference(sjets[i][elem].index)]

        for ev in evs: ev.sync()
    
    print("Assembling finished frame")
    framepieces = []
    for i in range(len(files)):
        tempframe = pd.DataFrame()
        for prop in jets[i]:
            tempframe[prop] = jets[i][prop][jets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        if option == "MuonEG" or option == "Parked":
            for prop in mus[i]:
                tempframe[f"m{prop}"] = mus[i][prop][mus[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        if "C" in stage:
            for prop in sjets[i]:
                tempframe[f"s1{prop}"] = sjets[i][prop][sjets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
                tempframe[f"s2{prop}"] = sjets[i][prop][sjets[i]['pt'].rank(axis=1,method='first',ascending=False) == 2].max(axis=1)
        tempframe['trigA'] = np.logical_and(l1s[i].SingleJet180[1],hlts[i].AK8PFJet500[1])
        tempframe['trigB'] = np.logical_and(l1s[i].SingleJet180[1],hlts[i].AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4[1])
        tempframe['trigC'] = np.logical_and(
            np.logical_or(l1s[i].DoubleJet112er2p3_dEta_Max1p6[1],l1s[i].DoubleJet150er2p5[1])
            ,hlts[i].DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71[1])
        tempframe['trigAB'] = np.logical_or(tempframe['trigA'],tempframe['trigB'])
        tempframe['trigABC']= np.logical_or(tempframe['trigAB'],tempframe['trigC'])
        # tempframe['L1_SingleJet180'] = l1s[i].SingleJet180[1]
        # tempframe['HLT_AK8PFJet500'] = hlts[i].AK8PFJet500[1]
        # tempframe['HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4'] = hlts[i].AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4[1]
        # tempframe['L1_DoubleJet112er2p3_dEta_Max1p6'] = l1s[i].DoubleJet112er2p3_dEta_Max1p6[1]
        # tempframe['L1_DoubleJet150er2p5'] = l1s[i].DoubleJet150er2p5[1]
        # tempframe['HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71'] = hlts[i].DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71[1]
        framepieces.append(tempframe)
    mergedframe = pd.concat(framepieces, ignore_index=True)
    mergedframe = mergedframe.dropna()
    
    
    if conf == "GGH_HPT.json":
        sigweight = (3.9 - 0.4*np.log2(mergedframe['pt']))
        sigweight[sigweight < 0.1] = 0.1
        mergedframe['extweight'] = mergedframe['extweight'] * sigweight
    
    if option == 'Parked' and not isdata: 
        mergedframe['extweight'] = lumipucalc(mergedframe)
            
    # pickle.dump(evs,open('effevents.p','wb'))
    # pickle.dump(mergedframe,open('effframe.p','wb'))
            
    
    print("Producing histograms")
    effplot = {}
    if stage == "A":
        denplot['pt'].fill(mergedframe['pt'],mergedframe['extweight'])
        numplot['pt'].fill(mergedframe['pt'][mergedframe['trigA'] == 1],
                     mergedframe['extweight'][mergedframe['trigA'] == 1])
        effplot.update({'pt': numplot['pt'].divideby(denplot['pt'],split=True,errmethod='effnorm')})
    
    if stage == "B":
        denplot['pt'].fill(mergedframe['pt'],mergedframe['extweight'])
        numplot['pt'].fill(mergedframe['pt'][mergedframe['trigB'] == 1],
                 mergedframe['extweight'][mergedframe['trigB'] == 1])
        effplot.update({'pt': numplot['pt'].divideby(denplot['pt'],split=True,errmethod='effnorm')})
        
        ## Trim down events to only passing values after the pt plateu, for further studies
        mergedframe = mergedframe[mergedframe['pt'] > 400]
        
        denplot['msoft'].fill(mergedframe['msoft'],mergedframe['extweight'])
        numplot['msoft'].fill(mergedframe['msoft'][mergedframe['trigB'] == 1],
                     mergedframe['extweight'][mergedframe['trigB'] == 1])
        effplot.update({'msoft': numplot['msoft'].divideby(denplot['msoft'],split=True,errmethod='effnorm')})
        
        denplot['DDBvL'].fill(mergedframe['DDBvL'],mergedframe['extweight'])
        numplot['DDBvL'].fill(mergedframe['DDBvL'][mergedframe['trigB'] == 1],
                     mergedframe['extweight'][mergedframe['trigB'] == 1])
        effplot.update({'DDBvL': numplot['DDBvL'].divideby(denplot['DDBvL'],split=True,errmethod='effnorm')})
    
    if stage == "C":
        tempframe = mergedframe[np.logical_and(mergedframe['s1pt'] > 140, mergedframe['s2pt'] > 140)]
        tempframe = tempframe[np.logical_and(tempframe['s1btagDeepB'] > .4184,tempframe['s2btagDeepB'] > .4184)]
        denplot['pt'].fill(tempframe['pt'],tempframe['extweight'])
        numplot['pt'].fill(tempframe['pt'][mergedframe['trigC'] == 1],
            tempframe['extweight'][mergedframe['trigC'] == 1])
        effplot.update({'pt': numplot['pt'].divideby(denplot['pt'],split=True,errmethod='effnorm')})
        
        tempframe = mergedframe[mergedframe['s1pt'] > mergedframe['s2pt']]
        tempframe = tempframe[np.logical_and(tempframe['s1btagDeepB'] > .4184,tempframe['s2btagDeepB'] > .4184)]
        denplot['s2pt'].fill(tempframe['s2pt'],tempframe['extweight'])
        numplot['s2pt'].fill(tempframe['s2pt'][mergedframe['trigC'] == 1],
            tempframe['extweight'][mergedframe['trigC'] == 1])
        tempframe = mergedframe[mergedframe['s1pt'] <= mergedframe['s2pt']]
        tempframe = tempframe[np.logical_and(tempframe['s1btagDeepB'] > .4184,tempframe['s2btagDeepB'] > .4184)]
        denplot['s2pt'].fill(tempframe['s1pt'],tempframe['extweight'])
        numplot['s2pt'].fill(tempframe['s1pt'][mergedframe['trigC'] == 1],
            tempframe['extweight'][mergedframe['trigC'] == 1])
        effplot.update({'s2pt': numplot['s2pt'].divideby(denplot['s2pt'],split=True,errmethod='effnorm')})
        
        tempframe = mergedframe[mergedframe['s1btagDeepB'] > mergedframe['s2btagDeepB']]
        tempframe = tempframe[np.logical_and(tempframe['s1pt'] > 150, tempframe['s2pt'] > 150)]
        denplot['lowb'].fill(tempframe['s2btagDeepB'],tempframe['extweight'])
        numplot['lowb'].fill(tempframe['s2btagDeepB'][mergedframe['trigC'] == 1],
            tempframe['extweight'][mergedframe['trigC'] == 1])
        tempframe = mergedframe[mergedframe['s1btagDeepB'] <= mergedframe['s2btagDeepB']]
        tempframe = tempframe[np.logical_and(tempframe['s1pt'] > 150, tempframe['s2pt'] > 150)]
        denplot['lowb'].fill(tempframe['s1btagDeepB'],tempframe['extweight'])
        numplot['lowb'].fill(tempframe['s1btagDeepB'][mergedframe['trigC'] == 1],
            tempframe['extweight'][mergedframe['trigC'] == 1])
        effplot.update({'lowb': numplot['lowb'].divideby(denplot['lowb'],split=True,errmethod='effnorm')})
    
    if stage == "D":
        denplot['pt'].fill(mergedframe['pt'],mergedframe['extweight'])
        numplot['pt'].fill(mergedframe['pt'][mergedframe['trigC'] == 1],
            mergedframe['extweight'][mergedframe['trigC'] == 1])
        effplot.update({'pt': numplot['pt'].divideby(denplot['pt'],split=True,errmethod='effnorm')})
        
    if len(stage) > 1:
        if stage == "CX": stage = "C"
        denplot['pt'].fill(mergedframe['pt'],mergedframe['extweight'])
        numplot['pt'].fill(mergedframe['pt'][mergedframe[f"trig{stage}"] == 1],
            mergedframe['extweight'][mergedframe[f"trig{stage}"] == 1])
        denplot['msoft'].fill(mergedframe['msoft'],mergedframe['extweight'])
        numplot['msoft'].fill(mergedframe['msoft'][mergedframe[f"trig{stage}"] == 1],
            mergedframe['extweight'][mergedframe[f"trig{stage}"] == 1])
        denplot['DDBvL'].fill(mergedframe['DDBvL'],mergedframe['extweight'])
        numplot['DDBvL'].fill(mergedframe['DDBvL'][mergedframe[f"trig{stage}"] == 1],
            mergedframe['extweight'][mergedframe[f"trig{stage}"] == 1])
        if "C" in stage:
            tempframe = mergedframe[mergedframe['s1pt'] > mergedframe['s2pt']]
            denplot['s2pt'].fill(tempframe['s2pt'],tempframe['extweight'])
            numplot['s2pt'].fill(tempframe['s2pt'][tempframe[f"trig{stage}"] == 1],
                tempframe['extweight'][mergedframe[f"trig{stage}"] == 1])
            tempframe = mergedframe[mergedframe['s1pt'] <= mergedframe['s2pt']]
            denplot['s2pt'].fill(tempframe['s1pt'],tempframe['extweight'])
            numplot['s2pt'].fill(tempframe['s1pt'][tempframe[f"trig{stage}"] == 1],
                tempframe['extweight'][mergedframe[f"trig{stage}"] == 1])
            
            tempframe = mergedframe[mergedframe['s1btagDeepB'] > mergedframe['s2btagDeepB']]
            denplot['s2deepb'].fill(tempframe['s2btagDeepB'],tempframe['extweight'])
            numplot['s2deepb'].fill(tempframe['s2btagDeepB'][tempframe[f"trig{stage}"] == 1],
                tempframe['extweight'][mergedframe[f"trig{stage}"] == 1])
            tempframe = mergedframe[mergedframe['s1btagDeepB'] <= mergedframe['s2btagDeepB']]
            denplot['s2deepb'].fill(tempframe['s1btagDeepB'],tempframe['extweight'])
            numplot['s2deepb'].fill(tempframe['s1btagDeepB'][tempframe[f"trig{stage}"] == 1],
                tempframe['extweight'][mergedframe[f"trig{stage}"] == 1])
        if stage == "C": stage = "CX"
        for p in denplot:
            effplot.update({p:numplot[p].divideby(denplot[p],split=True,errmethod='effnorm')})
        pickle.dump(effplot,open(f"Seffplots/{name}_{stage}_{option}_ScaleFactor.p",'wb'))
    else:
        pickle.dump(effplot,open(f"Effplots/{name}_EfficiencyPlot_{option}_{stage}.p",'wb'))
    for p in effplot:
        effplot[p].plot(htype='err')
    pickle.dump(effplot,open(f"Effplots/{name}_EfficiencyPlot_{option}_{stage}.p",'wb'))
    #sys.exit()
    # pickle.dump({'numplot':numplot,'denplot':denplot,'effplot':effplot},open('effplots.p','wb'))
#%%

def overlay(stage):
    #%%
    if len(stage) > 1:
        starr = ["AB","ABC","CX"]
        qcd,dt,rt = {},{},{}
        for st in starr:
            qcd.update({st: pickle.load(open(f"Seffplots/bGen+bEnr_{st}_Parked_ScaleFactor.p",'rb'))})
            dt.update( {st: pickle.load(open(f"Seffplots/ParkedSkim_{st}_Parked_ScaleFactor.p",'rb'))})
            rt.update({st:{}})
            for plot in qcd[st]:
                rt[st].update( {plot: dt[st][plot].divideby(qcd[st][plot],split=True,errmethod=None)})
                rt[st][plot].fname=f"Seffplots/FinalSF_{plot}_{st}"
                rt[st][plot].title=f"Final Scale Factor {st}"
                lower, upper = [],[]
                for i in range(len(rt[st][plot][0])):
                    C = rt[st][plot][0][i]
                    A = dt[st][plot][0][i]
                    B = qcd[st][plot][0][i]
                    dA, dB = [],[]
                    dA = np.sqrt((dt[st][plot].ser[0][i], dt[st][plot].ser[1][i]))
                    dB = np.sqrt((qcd[st][plot].ser[0][i],qcd[st][plot].ser[1][i]))
                    lower.append(np.power(C,2) * (np.power(dA[0]/A,2) + np.power(dB[1]/B,2)))
                    upper.append(np.power(C,2) * (np.power(dA[1]/A,2) + np.power(dB[0]/B,2)))
                    rt[st][plot].ser = np.array([lower,upper])
                rt[st][plot].plot(htype='err')
            rt.update({"meta":{"x":['AB','ABC','CX'],"y":['pt','msoft','DDBvL','s2pt','s2deepb']}})
            pickle.dump(rt, open(f"ScaleTensor",'wb'))
        sys.exit()
    print(f"Merging {stage} plots...")
    pqq = [pickle.load(open(f"Effplots/ParkedSkim_EfficiencyPlot_Parked_{stage}.p",'rb')),
           pickle.load(open(f"Effplots/bGen+bEnr_EfficiencyPlot_Parked_{stage}.p",'rb')),
           pickle.load(open(f"Effplots/bGen+bEnr_EfficiencyPlot__{stage}.p",'rb'))]
    pgg = [pickle.load(open(f"Effplots/ParkedSkim_EfficiencyPlot_Parked_{stage}.p",'rb')),
           pickle.load(open(f"Effplots/GGH_HPT_EfficiencyPlot_Parked_{stage}.p",'rb')),
           pickle.load(open(f"Effplots/GGH_HPT_EfficiencyPlot__{stage}.p",'rb'))]
    qtg = [pickle.load(open(f"Effplots/bGen+bEnr_EfficiencyPlot__{stage}.p",'rb')),
           pickle.load(open(f"Effplots/TTbar_EfficiencyPlot__{stage}.p",'rb')),
           pickle.load(open(f"Effplots/GGH_HPT_EfficiencyPlot__{stage}.p",'rb'))]
    
    pqql = ["Parking BPH Skim","QCD (Parked Weights)","Combined QCD"]
    pggl = ["Parking BPH Skim","ggH MC (Parked Weights)","ggH MC"]
    qtgl = ["Combined QCD","TTbar","ggH MC"]
    
    plt.clf()
    pqq[0]['pt'].make(htype='err',color='r')
    pqq[1]['pt'].make(htype='err',color='g')
    pqq[2]['pt'].fname=f"Effplots/pqqOverlay_{stage}"
    pqq[2]['pt'].plot(same=True,htype='err',legend=pqql)
    plt.clf()
    pgg[0]['pt'].make(htype='err',color='r')
    pgg[1]['pt'].make(htype='err',color='g')
    pgg[2]['pt'].fname=f"Effplots/pggOverlay_{stage}"
    pgg[2]['pt'].plot(same=True,htype='err',legend=pggl)
    plt.clf()
    qtg[0]['pt'].make(htype='err',color='r')
    qtg[1]['pt'].make(htype='err',color='g')
    qtg[2]['pt'].fname=f"Effplots/qtgOverlay_{stage}"
    qtg[2]['pt'].plot(same=True,htype='err',legend=qtgl)

    ##
    parked = pqq[0]['pt']
    qcdp = pqq[1]['pt']
    if stage == "A":   ptcut = 550
    elif stage == "B": ptcut = 400
    elif stage == "C": ptcut = 500
    
    # size = parked.size/(parked.bounds[1]-parked.bounds[0])*(parked.bounds[1]-ptcut)
    # bounds = (ptcut,parked.bounds[1])
    # parked.size = size
    # parked.bounds = bounds
    # qcdp.size = size
    # qcdp.bounds = bounds
    
    parked[0] = parked[0][parked[1][:-1] >= ptcut] 
    nlow = parked.ser[0][parked[1][:-1] >= ptcut]
    nup = parked.ser[1][parked[1][:-1] >= ptcut]
    parked.ser = np.array([nlow,nup])
    parked[1] = parked[1][parked[1] >= ptcut]
    
    qcdp[0] = qcdp[0][qcdp[1][:-1] >= ptcut]
    nlow = qcdp.ser[0][qcdp[1][:-1] >= ptcut]
    nup = qcdp.ser[1][qcdp[1][:-1] >= ptcut]
    qcdp.ser = np.array([nlow,nup])
    qcdp[1] = qcdp[1][qcdp[1] >= ptcut]
    
    effrat = parked.divideBy(qcdp,split=True,errmethod=None)
    for i in range(len(effrat.ser[0])):
        effrat.ser[0][i] = (parked.ser[0][i]/parked[0][i]) + (qcdp.ser[1][i]/qcdp[0][i])
        effrat.ser[1][i] = (parked.ser[1][i]/parked[0][i]) + (qcdp.ser[0][i]/qcdp[0][i])
    effrat.title = f"Efficiency scale factor {stage}"
    effrat.fname = f"Effplots/SFefficiency_{stage}"
    effrat.plot(htype='err')
    print("Merge done.")
#%%
def main():
    file, option, stage = '','',''
    merge=False
    full =False
    over =False
    letters = ["-A","-B","-C","-D","-AB","-ABC","-CX"]
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-f':
            file = sys.argv[i+1]
        elif sys.argv[i] == '-parked':
            option = 'Parked'
        elif sys.argv[i] == '-muoneg':
            option = 'MuonEG'
        elif sys.argv[i] in letters:
            stage = sys.argv[i][1:]
        elif sys.argv[i] == '-merge':
            merge=True
        elif sys.argv[i] == '-full':
            full =True
        elif sys.argv[i] == '-over':
            over =True
    if full:
        compare(file,option,"A")
        compare(file,option,"B")
        compare(file,option,"C")
        compare(file,option,"AB")
        compare(file,option,"ABC")
        compare(file,option,"CX")
    elif over:
        overlay("A")
        overlay("B")
        overlay("C")
        overlay("ABC")
    elif (file and stage) and not merge:
        compare(file,option,stage)
    elif merge and stage:
        overlay(stage)
    else:
        print("Expected arguments of the form: trigeff.py -f <config.json>")
        print("Options such as -parked and -muoneg can also be used to specify cuts")
        print("Specify pathway A,B,C, or D with an argument like -A,")
        print("or specify tensor production with -AB, -ABC, or -CX")
        
if __name__ == "__main__":
    main()