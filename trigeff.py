#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:42:03 2021

"""
import numpy as np
from analib import inc, Hist, PhysObj, Event
import pandas as pd
import uproot, json, sys, pickle

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

#%%
def compare(conf,option):
#%%    
    print(f"Analysing {conf} with {option} cuts")
    with open(conf) as f:
        confd = json.load(f)
        islhe =     confd['islhe']
        #isdata =    confd['isdata']
        files =     confd['files']
        if type(files) != list:
            files = [files]
        fweights =  confd['weight']
        if type(fweights) != list:
            fweights = [fweights]
        name =      confd['name']
        
    numplot = Hist(27,(150,1500),'pT of highest FatJet after cuts')
    denplot = Hist(27,(150,1500),'pT of highest FatJet after cuts and triggers')
    
    elecvars = ['pt','eta','mvaFall17V2Iso_WP90']
    muvars = ['pt','eta','mediumPromptId','miniIsoId','softId','dxy','dxyErr','ip3d']
    l1vars = ['SingleJet180','Mu7_EG23er2p5','Mu7_LooseIsoEG20er2p5','Mu20_EG10er2p5','SingleMu22',
              'SingleMu25']
    hltvars = ['AK8PFJet500','Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ','Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
               'Mu27_Ele37_CaloIdL_MW','Mu37_Ele27_CaloIdL_MW']
    
    print("Collecting event information")
    
    events, jets, elecs, mus, l1s, hlts = [],[],[],[],[],[]
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
        
    evs = []
    for i in range(len(files)):
        evs.append(Event(jets[i],elecs[i],mus[i],l1s[i],hlts[i]))
        
    

        
    for jet in jets:
        jet.cut(jet.pt > 170)
        jet.cut(abs(jet.eta)<2.4)
        jet.cut(jet.DDBvL > 0.8)
        jet.cut(jet.DeepB > 0.4184)
        jet.cut(jet.msoft > 90)
        jet.cut(jet.mass > 90)
        jet.cut(jet.msoft < 200)
        jet.cut(jet.npvsG >= 1)
        
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
            mu.cut(mu.eta < 2.4)
            mu.cut(mu.ip > 2)
            mu.cut(mu.ip3d < 0.5)
        #else: raise(NameError("Dataset name does not match expected"))
        
    for ev in evs: ev.sync()
    
    if option == 'MuonEG':
        for i in range(len(files)):
            l1s[i].cut(np.logical_and.reduce((
                np.logical_and(np.logical_or(l1s[i].Mu7_EG23er2p5,l1s[i].Mu7_LooseIsoEG20er2p5),hlts[i].Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ),
                np.logical_and(np.logical_or(l1s[i].Mu20_EG10er2p5,l1s[i].SingleMu22),hlts[i].Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL),
                np.logical_and(l1s[i].SingleMu25,hlts[i].Mu27_Ele37_CaloIdL_MW),
                np.logical_and(l1s[i].SingleMu25,hlts[i].Mu37_Ele27_CaloIdL_MW))))
    
            ## Makes a frame whose elements have the highest pt of the muon or electron in that position
            passf = elec[i].pt.combine(mu[i].pt,np.maximum,fill_value=0)
            ## Drops pt < 25
            passf = passf[passf > 25]
            ## Drops empty rows
            passf.dropma(how='all')
            ## The remaining events must have had an electron or muon with pt > 25 - the rest are removed
            elec.trimTo(passf)
        
    for ev in evs: ev.sync()
    print("Assembling finished frame")
    framepieces = []
    for i in range(len(files)):
        tempframe = pd.DataFrame()
        for prop in jets[i]:
            tempframe[prop] = jets[i][prop][jets[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        for prop in mus[i]:
            tempframe[f"m{prop}"] = mus[i][prop][mus[i]['pt'].rank(axis=1,method='first',ascending=False) == 1].max(axis=1)
        tempframe['L1_SingleJet180'] = l1s[i].SingleJet180[1]
        tempframe['HLT_AK8PFJet500'] = hlts[i].AK8PFJet500[1]
        framepieces.append(tempframe)
    mergedframe = pd.concat(framepieces, ignore_index=True)
    mergedframe = mergedframe.dropna()
    
    if conf == "GGH_HPT.json":
        mergedframe['extweight'] = mergedframe['extweight'] * (3.9 - 0.4*np.log2(mergedframe['pt']))
    
    if option == 'Parked':
        for i in range(len(files)):
            mergedframe['extweight'] = lumipucalc(mergedframe)
    print("Producing histograms")
    denplot.fill(mergedframe['pt'],mergedframe['extweight'])
    numplot.fill(mergedframe['pt'][np.logical_and(mergedframe['L1_SingleJet180'] == 1, mergedframe['HLT_AK8PFJet500'] == 1)],
                 mergedframe['extweight'][np.logical_and(mergedframe['L1_SingleJet180'] == 1, mergedframe['HLT_AK8PFJet500'] == 1)])
    effplot = numplot.divideby(denplot,split=True,trimnoise=.001,errmethod='effnorm')
    effplot.xlabel = 'AK8 jets passing cuts / passing cuts + triggers'
    effplot.ylabel = f"{option} Ratio"
    effplot.title  = name
    effplot.fname  = f"Effplots/{name}_EfficiencyPlot_{option}"
    effplot.ylim   = (0,1)
    effplot.plot(htype='err')
    pickle.dump(effplot,open(f"Effplots/{name}_EfficiencyPlot_{option}.p",'wb'))
    sys.exit()
#%%

def main():
    file, option = '',''
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-f':
            file = sys.argv[i+1]
        elif sys.argv[i] == '-parked':
            option = 'Parked'
        elif sys.argv[i] == 'muoneg':
            option = 'MuonEG'
    if file:
        compare(file,option)
    else:
        print("Expected arguments of the form: trigeff.py -f <config.json>")
        print("Options such as -parked and -muoneg can also be used to specify cuts")
        
if __name__ == "__main__":
    main()