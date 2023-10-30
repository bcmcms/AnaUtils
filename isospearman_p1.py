#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
### Compiled with Keras-2.3.1 Tensorflow-1.14.0                      ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

# from ROOT import TH1F, TFile, gROOT, TCanvas

import sys, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist as Hist, PhysObj, Event, inc, fstrip, InputConfig, dphi, Hist2d as Hist2D
from analib import dframe as DataFrame
import pickle
import copy as cp
#from uproot_methods import TLorentzVector, TLorentzVectorArray
#from sklearn.model_selection import train_test_split
#from tensorflow.python.keras import backend as BE
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform as squareform

import pdb

import mplhep as hep
# plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)
pd.set_option('mode.chained_assignment', None)

#evtlist = [35899001,24910172,106249475,126514437,43203653,27186346,17599588,64962950,61283040,54831588]



#%%

# useF = True
nbin = 17
RATIO = False
BLIND = True
path="Snetplots"
refpath="Snetplots"
REGION=''
if REGION: 
    subpath = REGION
    BLIND = False
else: subpath="BaseU"
dictname='isospearmandictBq'
folders = [f"{path}/Full/",f"{path}/QCDEnr/",f"{path}/QCDGen/",f"{path}/QCDInc/",
           f"{path}/TTbar/",f"{path}/WJets/",f"{path}/ZJets/",f"{path}/Data/"]
bgslice = ['Combined QCD','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
namelist = ['Full','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
sysvars = ['pt','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','submass1','submass2','nsv','subtau1','subtau2','n2b1','Trig']

isospearmandict = {
    "nbin": nbin,
    "BLIND":BLIND,
    "path": path,
    "rpath":refpath,
    "REGION":REGION
    }
# folders = ["Diststore/QCDGen/"]
## Create ROOT file for Combine
# os.remove(f"SubNet_CombinedX{nbin}.root")

## controls weighting amount for shape uncertainties
wval = 1.1


Adat = pickle.load(open(f"{refpath}/BaseU/GGH_HPT vs Combined QCD saveballA.p",'rb'))
Bdat = pickle.load(open(f"{refpath}/BaseU/GGH_HPT vs Combined QCD saveballB.p",'rb'))
Cdat = pickle.load(open(f"{refpath}/BaseU/GGH_HPT vs Combined QCD saveballC.p",'rb'))
Fdat = pickle.load(open(f"{refpath}/BaseU/GGH_HPT vs Combined QCD saveballF.p",'rb'))
for elem in [Adat, Bdat, Cdat, Fdat]:
    elem["SensB"] = [0] + list(elem["SensB"]) + [1]
    elem["SensS"] = list(elem["SensS"])

    for i in range(1,6):
        elem["SensB"].pop(i)
        elem["SensS"].pop(i)
ASensS = Adat["SensS"]
BSensS = Bdat["SensS"]
CSensS = Cdat["SensS"]
FSensS = Fdat["SensS"]
ASensB = Adat["SensB"]
BSensB = Bdat["SensB"]
CSensB = Cdat["SensB"]
FSensB = Fdat["SensB"]
## So far A, B, C aren't used further. They are emptied here to prevent future misreference
del(Adat, Bdat, Cdat)

for bg in bgslice:
    if 'Data' in bg:
        sf = 'JetHT'
        bf = 'Combined QCD'
    else:
        sf = 'GGH_HPT'
        bf = bg
    sigframe, bgframe = DataFrame(), DataFrame()
    sigframe['A'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} diststtA.p",'rb'))[:,0]
    sigframe['B'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} diststtB.p",'rb'))
    sigframe['C'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} diststtC.p",'rb'))[:,0]
    bgframe['A'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} distbttA.p",'rb'))[:,0]
    bgframe['B'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} distbttB.p",'rb'))
    bgframe['C'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} distbttC.p",'rb'))[:,0]

    sigframe['F'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} diststtF.p",'rb'))[:,0]
    bgframe['F'] = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} distbttF.p",'rb'))[:,0]

    Fdat = pickle.load(open(f"{path}/{subpath}/{sf} vs {bf} saveballF.p",'rb'))       

    # sbounds = FSensS + [x + 1 for x in FSensS[1:]] + [x + 2 for x in FSensS[1:]] +\
    #         [x + 3 for x in FSensS[1:]] + [x + 4 for x in FSensS[1:]]
    asbounds = ASensS + [x + 1 for x in ASensS[1:]] + [x + 2 for x in ASensS[1:]] +\
            [x + 3 for x in ASensS[1:]] + [x + 4 for x in ASensS[1:]]
    bsbounds = BSensS + [x + 1 for x in BSensS[1:]] + [x + 2 for x in BSensS[1:]] +\
            [x + 3 for x in BSensS[1:]] + [x + 4 for x in BSensS[1:]]
    csbounds = CSensS + [x + 1 for x in CSensS[1:]] + [x + 2 for x in CSensS[1:]] +\
            [x + 3 for x in CSensS[1:]] + [x + 4 for x in CSensS[1:]]
    # bbounds = FSensB + [x + 1 for x in FSensB[1:]] + [x + 2 for x in FSensB[1:]] +\
    #         [x + 3 for x in FSensB[1:]] + [x + 4 for x in FSensB[1:]]
    abbounds = ASensB + [x + 1 for x in ASensB[1:]] + [x + 2 for x in ASensB[1:]] +\
            [x + 3 for x in ASensB[1:]] + [x + 4 for x in ASensB[1:]]
    bbbounds = BSensB + [x + 1 for x in BSensB[1:]] + [x + 2 for x in BSensB[1:]] +\
            [x + 3 for x in BSensB[1:]] + [x + 4 for x in BSensB[1:]]
    cbbounds = CSensB + [x + 1 for x in CSensB[1:]] + [x + 2 for x in CSensB[1:]] +\
            [x + 3 for x in CSensB[1:]] + [x + 4 for x in CSensB[1:]]
    # raise ValueError("test")
    if   bg == 'Combined QCD': fn = 'Full'
    elif bg == 'bEnriched': fn = 'QCDEnr'
    elif bg == 'bGen': fn = 'QCDGen'
    elif bg == 'bInc': fn = 'QCDInc'
    else: fn = bg
    distplots = {
        "sFA": Hist2D([FSensS,ASensB],None,"Full Network Signal","Subnet A signal",f"ratio/{fn}_FAsubdistS"),
        "sFB": Hist2D([FSensS,BSensB],None,"Full Network Signal","Subnet B signal",f"ratio/{fn}_FBsubdistS"),
        "sFC": Hist2D([FSensS,CSensB],None,"Full Network Signal","Subnet C signal",f"ratio/{fn}_FCsubdistS"),
        "bFA": Hist2D([FSensS,ASensB],None,"Full Network BG","Subnet A BG",f"ratio/{fn}_FAsubdistB"),
        "bFB": Hist2D([FSensS,BSensB],None,"Full Network BG","Subnet B BG",f"ratio/{fn}_FBsubdistB"),
        "bFC": Hist2D([FSensS,CSensB],None,"Full Network BG","Subnet C BG",f"ratio/{fn}_FCsubdistB"),
        "bAB": Hist2D([ASensB,BSensB],None,"Subnet A BG","Subnet B BG",f"ratio/{fn}_ABsubdistB"),
        "bAC": Hist2D([ASensB,CSensB],None,"Subnet A BG","Subnet C BG",f"ratio/{fn}_ACsubdistB"),
        "bBC": Hist2D([BSensB,CSensB],None,"Subnet B BG","Subnet C BG",f"ratio/{fn}_BCsubdistB"),
        }
    flatplots = {
        "sFA": Hist(abbounds,None,"20%-quantile A occupancy","Signal Events","FAflatS",f"ratio/{fn}_FA_signal"),
        "sFB": Hist(bbbounds,None,"20%-quantile B occupancy","Signal Events","FBflatS",f"ratio/{fn}_FB_signal"),
        "sFC": Hist(cbbounds,None,"20%-quantile C occupancy","Signal Events","FCflatS",f"ratio/{fn}_FC_signal"),
        "sAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Signal Events",f"ratio/{fn}_ABflatS","BinA"),
        "sAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Signal Events",f"ratio/{fn}_ACflatS","CinA"),
        "sBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Signal Events",f"ratio/{fn}_BCflatS","CinB"),
        "bFA": Hist(abbounds,None,"20%-quantile A occupancy","Background Events",f"ratio/{fn}_FAflatB","NetA"),
        "bFB": Hist(bbbounds,None,"20%-quantile B occupancy","Background Events",f"ratio/{fn}_FBflatB","NetB"),
        "bFC": Hist(cbbounds,None,"20%-quantile C occupancy","Background Events",f"ratio/{fn}_FCflatB","NetC"),
        "bAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Background Events",f"ratio/{fn}_ABflatB","BinA"),
        "bAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Background Events",f"ratio/{fn}_ACflatB","CinA"),
        "bBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Background Events",f"ratio/{fn}_BCflatB","CinB"),
        }

    sigframe["W"] = Fdat["WS"].reset_index(drop=True)
    bgframe["W"] = Fdat["WB"].reset_index(drop=True)
    # sigframe["Abin"], sigframe["Bbin"], sigframe["Cbin"] = 5, 5, 5
    # bgframe["Abin"], bgframe["Bbin"], bgframe["Cbin"] = 5, 5, 5
    sigframe ["Fbin"], bgframe["Fbin"] = 5, 5
    bgframe["Abin"], bgframe["Bbin"], bgframe["Cbin"] = 5, 5, 5
    sigframe["Abin"], sigframe["Bbin"], sigframe["Cbin"] = 5, 5, 5
    for i in range(5,0,-1):
        sigframe["Abin"][sigframe["A"] <= ASensB[i]] = i-1
        sigframe["Bbin"][sigframe["B"] <= BSensB[i]] = i-1
        sigframe["Cbin"][sigframe["C"] <= CSensB[i]] = i-1
        sigframe["Fbin"][sigframe["F"] <= FSensS[i]] = i-1
        bgframe["Abin"][bgframe["A"] <= ASensB[i]] = i-1
        bgframe["Bbin"][bgframe["B"] <= BSensB[i]] = i-1
        bgframe["Cbin"][bgframe["C"] <= CSensB[i]] = i-1
        bgframe["Fbin"][bgframe["F"] <= FSensS[i]] = i-1
        
    # ## Is this blinding code?
    # if BLIND:
    #     bgframe = bgframe[~((bgframe["Abin"] >= 3) & (bgframe["Bbin"] >= 3))]
    #     bgframe = bgframe[~((bgframe["Abin"] >= 3) & (bgframe["Cbin"] >= 3))]
    #     bgframe = bgframe[~((bgframe["Bbin"] >= 3) & (bgframe["Cbin"] >= 3))]
    #     sigframe = sigframe[~((sigframe["Abin"] >= 3) & (sigframe["Bbin"] >= 3))]
    #     sigframe = sigframe[~((sigframe["Abin"] >= 3) & (sigframe["Cbin"] >= 3))]
    #     sigframe = sigframe[~((sigframe["Bbin"] >= 3) & (sigframe["Cbin"] >= 3))]

    distplots["sFA"].fill(sigframe["F"],sigframe["A"],sigframe['W'])
    distplots["sFB"].fill(sigframe["F"],sigframe["B"],sigframe['W'])
    distplots["sFC"].fill(sigframe["F"],sigframe["C"],sigframe['W'])
    distplots["bFA"].fill(bgframe["F"],bgframe["A"],bgframe['W'])
    distplots["bFB"].fill(bgframe["F"],bgframe["B"],bgframe['W'])
    distplots["bFC"].fill(bgframe["F"],bgframe["C"],bgframe['W'])
    distplots["bAB"].fill(bgframe["A"],bgframe["B"],bgframe['W'])
    distplots["bAC"].fill(bgframe["A"],bgframe["C"],bgframe['W'])
    distplots["bBC"].fill(bgframe["B"],bgframe["C"],bgframe['W'])


    for p in distplots:
        distplots[p][1] = [0,1,2,3,4,5]
        distplots[p][2] = [0,1,2,3,4,5]
        distplots[p][0] /= (distplots[p][0].sum()/25)
        distplots[p].plot(text=True,edgecolor='k',tlen=5)

    if fn == 'Full':
        if True:#(path != refpath) and (not REGION):
            ABCtensor = pickle.load(open(f"{refpath}/ABCtensor.p",'rb'))
            fdistplots = pickle.load(open(f"{refpath}/fdistplots.p",'rb'))
        else:
            fdistplots = cp.deepcopy(distplots)
            fbgframe = cp.deepcopy(bgframe)
            fflatplots = cp.deepcopy(flatplots)
            
            ABCtensor = {}
            for i in range(5):
                ABCtensor.update({i:cp.deepcopy(distplots["bAB"])})
            tsum = 0
            for i in range(5):
                ABCtensor[i].fill(bgframe["A"][bgframe["Cbin"] == i],bgframe["B"][bgframe["Cbin"] == i],bgframe['W'][bgframe["Cbin"] == i])
                tsum += ABCtensor[i][0].sum()
            for i in range(5):
                ABCtensor[i][0] /= (tsum / 125)
            pickle.dump(ABCtensor, open(f"{refpath}/ABCtensor.p",'wb'))
            pickle.dump(fdistplots,open(f"{refpath}/fdistplots.p",'wb'))
        varsysdict = {}
        if not REGION:
            for v in sysvars:
                varsysdict.update({v:{
                    "A":{},
                    "B":{},
                    "C":{}
                    }})
                for d in ["Up","Down"]:
                    varsysdict[v]["A"].update({d: Hist(abbounds,None,"20%-quantile A occupancy","Signal Events",f"FA {v} {d}",f"ratio/{v}_FA_{d}")})
                    varsysdict[v]["B"].update({d: Hist(bbbounds,None,"20%-quantile B occupancy","Signal Events",f"FB {v} {d}",f"ratio/{v}_FB_{d}")})
                    varsysdict[v]["C"].update({d: Hist(cbbounds,None,"20%-quantile C occupancy","Signal Events",f"FC {v} {d}",f"ratio/{v}_FC_{d}")})
    
                    sysframe =  DataFrame()
                    sysframe['A'] = pickle.load(open(f"{path}/{v}{d}/{sf} vs {bf} diststtA.p",'rb'))[:,0]
                    sysframe['B'] = pickle.load(open(f"{path}/{v}{d}/{sf} vs {bf} diststtB.p",'rb'))
                    sysframe['C'] = pickle.load(open(f"{path}/{v}{d}/{sf} vs {bf} diststtC.p",'rb'))[:,0]
                    sysframe['F'] = pickle.load(open(f"{path}/{v}{d}/GGH_HPT vs Combined QCD diststtF.p",'rb'))[:,0]
                    Fsysdat = pickle.load(open(f"{path}/{v}{d}/GGH_HPT vs Combined QCD saveballF.p",'rb'))   
                    
                    sysframe["W"] = Fsysdat["WS"].reset_index(drop=True)
                    sysframe["Fbin"], sysframe["Abin"], sysframe["Bbin"], sysframe["Cbin"] = 5, 5, 5, 5
                    for i in range(5,0,-1):
                        sysframe["Abin"][sysframe["A"] <= ASensB[i]] = i-1
                        sysframe["Bbin"][sysframe["B"] <= BSensB[i]] = i-1
                        sysframe["Cbin"][sysframe["C"] <= CSensB[i]] = i-1
                        sysframe["Fbin"][sysframe["F"] <= FSensS[i]] = i-1
                        
                    # # is this blinding code?
                    # if BLIND:
                    #     sysframe = sysframe[~((sysframe["Abin"] >= 3) & (sysframe["Bbin"] >= 3))]
                    #     sysframe = sysframe[~((sysframe["Abin"] >= 3) & (sysframe["Cbin"] >= 3))]
                    #     sysframe = sysframe[~((sysframe["Bbin"] >= 3) & (sysframe["Cbin"] >= 3))]
                    for i in range(0,5):
                        varsysdict[v]["A"][d].fill(sysframe["A"][sysframe["Fbin"] == i] + i, sysframe["W"][sysframe["Fbin"] == i])
                        varsysdict[v]["B"][d].fill(sysframe["B"][sysframe["Fbin"] == i] + i, sysframe["W"][sysframe["Fbin"] == i])
                        varsysdict[v]["C"][d].fill(sysframe["C"][sysframe["Fbin"] == i] + i, sysframe["W"][sysframe["Fbin"] == i])
            

        


    bgframe['3DWU'] = 0
    bgframe['3DWD'] = 0
    for c in range(5):
        for b in range(5):
            for a in range(5):
                bgframe['3DWU'][(bgframe['Abin'] == a) & (bgframe['Bbin'] == b) & (bgframe['Cbin'] == c)] =   ABCtensor[c][0][a][b]
                bgframe['3DWD'][(bgframe['Abin'] == a) & (bgframe['Bbin'] == b) & (bgframe['Cbin'] == c)] = 1/ABCtensor[c][0][a][b]
    bgframe['3DWU'] *= bgframe['W']
    bgframe['3DWD'] *= bgframe['W']

    bounds = [abbounds, bbbounds, cbbounds]
    qdict = {"A":{},"B":{},"C":{}}
    net = ["A","B","C"]
    # debugdict = {}
    for n in range(3):
        for b in range(5):
            qdict[net[n]].update({b:{}})
            for key in flatplots:
                qdict[net[n]][b].update({key:{}})
            for s in net:
                qdict[net[n]][b].update({f"bF{s}3D":{}})
            for d in ['U','D']:
                tempwB, tempwS = cp.deepcopy(bgframe["W"]), cp.deepcopy(sigframe["W"])
                if d == 'U':
                    tempwS[sigframe[f"{net[n]}bin"] == b] *= wval
                    tempwB[bgframe[f"{net[n]}bin"] == b]  *= wval
                else: 
                    tempwS[sigframe[f"{net[n]}bin"] == b] /= wval
                    tempwB[bgframe[f"{net[n]}bin"] == b]  /= wval
                for key in flatplots:
                    qdict[net[n]][b][key].update({d:cp.deepcopy(flatplots[key])})
                for s in net:
                    qdict[net[n]][b][f"bF{s}3D"].update({d:cp.deepcopy(flatplots[f"bF{s}"])})
                for i in range(5):
                    qdict[net[n]][b]["bAB"][d].fill(bgframe["B"][bgframe["Abin"] == i] + i, tempwB[bgframe["Abin"] == i])
                    qdict[net[n]][b]["bAC"][d].fill(bgframe["C"][bgframe["Abin"] == i] + i, tempwB[bgframe["Abin"] == i])
                    qdict[net[n]][b]["bBC"][d].fill(bgframe["C"][bgframe["Bbin"] == i] + i, tempwB[bgframe["Bbin"] == i])
                    qdict[net[n]][b]["bFA"][d].fill(bgframe["A"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                    qdict[net[n]][b]["bFB"][d].fill(bgframe["B"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                    qdict[net[n]][b]["bFC"][d].fill(bgframe["C"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                    for s in net:
                        qdict[net[n]][b][f"bF{s}3D"][d].fill(bgframe[f"{s}"][bgframe["Fbin"] == i] + i, bgframe[f"3DW{d}"][bgframe["Fbin"] == i])
                        qdict[net[n]][b][f"bF{s}3D"][d].title = f"Net{s}_F3D"
                    # qdict[net[n]][b]["sFA"].fill(sigframe["A"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                    # qdict[net[n]][b]["sFB"].fill(sigframe["B"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                    # qdict[net[n]][b]["sFC"].fill(sigframe["C"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                    # qdict[net[n]][b]["sAB"].fill(sigframe["B"][bgframe["Abin"] == i] + i, tempwS[bgframe["Abin"] == i])
                    # qdict[net[n]][b]["sAC"].fill(sigframe["C"][bgframe["Abin"] == i] + i, tempwS[bgframe["Abin"] == i])
                    # qdict[net[n]][b]["sBC"].fill(sigframe["C"][bgframe["Bbin"] == i] + i, tempwS[bgframe["Bbin"] == i])
    vdict = {}
    for n in ["AB","AC","BC","FA","FB","FC"]:
        vdict.update({n:{}})
        for sub in ["AB","AC","BC"]:
            vdict[n].update({sub:{}})
            for l in sub:
                vdict[n][sub].update({l:{}})
                for b in range(5):
                    vdict[n][sub][l].update({b:{}})
                    for d in ['U','D']:
                        vdict[n][sub][l][b].update({d:cp.deepcopy(flatplots[f"b{n}"])})
                        for i in range(5):
                            tempwB = cp.deepcopy(bgframe["W"])
                            for j in range(5):
                                if l == sub[0]:
                                    if d == 'U': tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] *= fdistplots[f"b{sub}"][0][b][j]
                                    else:        tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] /= fdistplots[f"b{sub}"][0][b][j]
                                    # debugdict.update({f"{n}{sub}{l}{b}{j}":distplots[f"b{sub}"][0][j][b]})
                                else:
                                    if d == 'U': tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[0]}bin"] == j)] *= fdistplots[f"b{sub}"][0][j][b]
                                    else:        tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[0]}bin"] == j)] /= fdistplots[f"b{sub}"][0][j][b]
                                    # test[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[0]}bin"] == j)] *= distplots[f"b{sub}"][0][j][b]
                                    # debugdict.update({f"{n}{sub}{l}{b}{j}":fdistplots[f"b{sub}"][0][j][b]})
                            vdict[n][sub][l][b][d].fill(bgframe[f"{n[-1]}"][bgframe[f"{n[0]}bin"] == i] + i, tempwB[bgframe[f"{n[0]}bin"] == i])
                vdict[n][sub][l].update({"F" :{}})
                vdict[n][sub][l].update({"3D":{}})
                for d in ['U','D']:
                    test = cp.deepcopy(bgframe["W"])
                    # import pdb;
                    # pdb.set_trace()
                    for b in range(5):
                        for j in range(5):
                            if d == 'U': test[(bgframe[f"{sub[0]}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] *= fdistplots[f"b{sub}"][0][b][j]
                            else:        test[(bgframe[f"{sub[0]}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] /= fdistplots[f"b{sub}"][0][b][j]
                    vdict[n][sub][l]["F" ].update({d:cp.deepcopy(flatplots[f"b{n}"])})
                    vdict[n][sub][l]["3D"].update({d:cp.deepcopy(flatplots[f"b{n}"])})
                    for i in range(5):
                        vdict[n][sub][l]["F" ][d].fill(bgframe[f"{n[-1]}"][bgframe[f"{n[0]}bin"] == i] + i, test[bgframe[f"{n[0]}bin"] == i])
                        vdict[n][sub][l]["3D"][d].fill(bgframe[f"{n[-1]}"][bgframe[f"{n[0]}bin"] == i] + i, bgframe[f"3DW{d}"][bgframe[f"{n[0]}bin"] == i])

    for i in range(5):
        if True:#'Data' in fn:
            flatplots["sFA"].fill(sigframe["A"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 0)] + i, sigframe["W"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 0)])
            flatplots["sFB"].fill(sigframe["B"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 1)] + i, sigframe["W"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 1)])
            flatplots["sFC"].fill(sigframe["C"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 2)] + i, sigframe["W"][(sigframe["Fbin"] == i) & (sigframe.index%3 == 2)])
            flatplots["sAB"].fill(sigframe["B"][(sigframe["Abin"] == i) & (sigframe.index%3 == 1)] + i, sigframe["W"][(sigframe["Abin"] == i) & (sigframe.index%3 == 1)])
            flatplots["sAC"].fill(sigframe["C"][(sigframe["Abin"] == i) & (sigframe.index%3 == 2)] + i, sigframe["W"][(sigframe["Abin"] == i) & (sigframe.index%3 == 2)])
            flatplots["sBC"].fill(sigframe["C"][(sigframe["Bbin"] == i) & (sigframe.index%3 == 2)] + i, sigframe["W"][(sigframe["Bbin"] == i) & (sigframe.index%3 == 2)])
        else:
            flatplots["sFA"].fill(sigframe["A"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
            flatplots["sFB"].fill(sigframe["B"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
            flatplots["sFC"].fill(sigframe["C"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
            flatplots["sAB"].fill(sigframe["B"][sigframe["Abin"] == i] + i, sigframe["W"][sigframe["Abin"] == i])
            flatplots["sAC"].fill(sigframe["C"][sigframe["Abin"] == i] + i, sigframe["W"][sigframe["Abin"] == i])
            flatplots["sBC"].fill(sigframe["C"][sigframe["Bbin"] == i] + i, sigframe["W"][sigframe["Bbin"] == i])
        # flatplots["bFA"].fill(bgframe["A"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        # flatplots["bFB"].fill(bgframe["B"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        # flatplots["bFC"].fill(bgframe["C"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        # flatplots["bAB"].fill(bgframe["B"][bgframe["Abin"] == i] + i, bgframe["W"][bgframe["Abin"] == i])
        # flatplots["bAC"].fill(bgframe["C"][bgframe["Abin"] == i] + i, bgframe["W"][bgframe["Abin"] == i])
        # flatplots["bBC"].fill(bgframe["C"][bgframe["Bbin"] == i] + i, bgframe["W"][bgframe["Bbin"] == i])
        flatplots["bFA"].fill(bgframe["A"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 0)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 0)])
        flatplots["bFB"].fill(bgframe["B"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 1)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 1)])
        flatplots["bFC"].fill(bgframe["C"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)])
        flatplots["bAB"].fill(bgframe["B"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 1)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 1)])
        flatplots["bAC"].fill(bgframe["C"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)])
        flatplots["bBC"].fill(bgframe["C"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)] + i, bgframe["W"][(bgframe["Fbin"] == i) & (bgframe.index%3 == 2)])

#%%
    if nbin == 17:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
            # if not ('Data' in fn and 's' in p):
            #     flatplots[p].ndivide(3)
            flatplots[p].plot(htype='bar',logv=False,error=True)
            for d in ['U','D']:
                for n in net:
                    for b in qdict[n]:
                        if "F" in p:
                            for i in range(16,20):
                                qdict[n][b][p][d][0][15] += qdict[n][b][p][d][0][i]
                                qdict[n][b][p][d].ser[15] += qdict[n][b][p][d].ser[i]
                            qdict[n][b][p][d][0][16] *= 0
                            qdict[n][b][p][d].ser[16] *= 0
                            for i in range(20,25):
                                qdict[n][b][p][d][0][16] += qdict[n][b][p][d][0][i]
                                qdict[n][b][p][d].ser[16] += qdict[n][b][p][d].ser[i]
                            qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[17,18,19,20,21,22,23,24])
                            qdict[n][b][p][d].ser = np.delete(qdict[n][b][p][d].ser,[17,18,19,20,21,22,23,24])
                            qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-8,-7,-6,-5,-4,-3,-2,-1])
                            
                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                        else: 
                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                        qdict[n][b][p][d].ndivide(3)
            if "F" in p:
                for i in range(16,20):
                    flatplots[p][0][15] += flatplots[p][0][i]
                    flatplots[p].ser[15] += flatplots[p].ser[i]
                flatplots[p][0][16] *= 0
                flatplots[p].ser[16] *= 0
                for i in range(20,25):
                    flatplots[p][0][16] += flatplots[p][0][i]
                    flatplots[p].ser[16] += flatplots[p].ser[i]
                flatplots[p][0] = np.delete(flatplots[p][0],[17,18,19,20,21,22,23,24])
                flatplots[p].ser = np.delete(flatplots[p].ser,[17,18,19,20,21,22,23,24])
                flatplots[p][1] = np.delete(flatplots[p][1],[-8,-7,-6,-5,-4,-3,-2,-1])
        for d in ['U','D']:
            for p in ["bFA3D","bFB3D","bFC3D"]:
                for n in net:
                    for b in qdict[n]:
                        for i in range(16,20):
                            qdict[n][b][p][d][0][15] += qdict[n][b][p][d][0][i]
                            qdict[n][b][p][d].ser[15] += qdict[n][b][p][d].ser[i]
                        qdict[n][b][p][d][0][16] *= 0
                        qdict[n][b][p][d].ser[16] *= 0
                        for i in range(20,25):
                            qdict[n][b][p][d][0][16] += qdict[n][b][p][d][0][i]
                            qdict[n][b][p][d].ser[16] += qdict[n][b][p][d].ser[i]
                        qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[17,18,19,20,21,22,23,24])
                        qdict[n][b][p][d].ser = np.delete(qdict[n][b][p][d].ser,[17,18,19,20,21,22,23,24])
                        qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-8,-7,-6,-5,-4,-3,-2,-1])
                        for i in range(qdict[n][b][p][d][1].shape[0]):
                            qdict[n][b][p][d][1][i] = 0.2 * i
                        qdict[n][b][p][d].ndivide(3)
            for n in vdict:
                for s in vdict[n]:
                    for l in vdict[n][s]:
                        for b in vdict[n][s][l]:
                            vdict[n][s][l][b][d].ndivide(3)
                            if "F" in n:
                                for i in range(16,20):
                                    vdict[n][s][l][b][d][0][15] += vdict[n][s][l][b][d][0][i]
                                    vdict[n][s][l][b][d].ser[15] += vdict[n][s][l][b][d].ser[i]
                                vdict[n][s][l][b][d][0][16] *= 0
                                vdict[n][s][l][b][d].ser[16] *= 0
                                for i in range(20,25):
                                    vdict[n][s][l][b][d][0][16] += vdict[n][s][l][b][d][0][i]
                                    vdict[n][s][l][b][d].ser[16] += vdict[n][s][l][b][d].ser[i]
                                vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[17,18,19,20,21,22,23,24])
                                vdict[n][s][l][b][d].ser = np.delete(vdict[n][s][l][b][d].ser,[17,18,19,20,21,22,23,24])
                                vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-8,-7,-6,-5,-4,-3,-2,-1])
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
                            else:
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
        if ('Full' in fn) and (not REGION):
            for v in varsysdict:
                for s in varsysdict[v]:
                    for d in varsysdict[v][s]:
                        for i in range(varsysdict[v][s][d][1].shape[0]):
                            varsysdict[v][s][d][1][i] = 0.2*i
                        varsysdict[v][s][d].ndivide(3)
                        for i in range(16,20):
                            varsysdict[v][s][d][0][15] += varsysdict[v][s][d][0][i]
                            varsysdict[v][s][d].ser[15] += varsysdict[v][s][d].ser[i]
                        varsysdict[v][s][d][0][16] *= 0
                        varsysdict[v][s][d].ser[16] *= 0
                        for i in range(20,25):
                            varsysdict[v][s][d][0][16] += varsysdict[v][s][d][0][i]
                            varsysdict[v][s][d].ser[16] += varsysdict[v][s][d].ser[i]
                        varsysdict[v][s][d][0] = np.delete(varsysdict[v][s][d][0],[17,18,19,20,21,22,23,24])
                        varsysdict[v][s][d].ser = np.delete(varsysdict[v][s][d].ser,[17,18,19,20,21,22,23,24])
                        varsysdict[v][s][d][1] = np.delete(varsysdict[v][s][d][1],[-8,-7,-6,-5,-4,-3,-2,-1])
    elif nbin == 25:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
                # pass
            # if not ('Data' in fn and 's' in p):
            #     flatplots[p].ndivide(3)
            flatplots[p].plot(htype='bar',logv=False,error=True)
            for d in ['U','D']:
                for n in net:
                    for b in qdict[n]:
                        for i in range(qdict[n][b][p][d][1].shape[0]):
                            qdict[n][b][p][d][1][i] = 0.2 * i
                        qdict[n][b][p][d].ndivide(3)
        for d in ['U','D']:
            for p in ["bFA3D","bFB3D","bFC3D"]:
                for n in net:
                    for b in qdict[n]:
                        for i in range(qdict[n][b][p][d][1].shape[0]):
                            qdict[n][b][p][d][1][i] = 0.2 * i
                        qdict[n][b][p][d].ndivide(3)
            for n in vdict:
                for s in vdict[n]:
                    for l in vdict[n][s]:
                        for b in vdict[n][s][l]:
                            vdict[n][s][l][b][d].ndivide(3)
                            for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                vdict[n][s][l][b][d][1][i] = 0.2 * i
            if 'Full' in fn:
                for v in varsysdict:
                    for s in varsysdict[v]:
                        for d in varsysdict[v][s]:
                            for i in range(varsysdict[v][s][d].shape[0]):
                                varsysdict[v][s][d][1][i] = 0.2*i
                            varsysdict[v][s][d].ndivide(3)
                    
    else: raise ValueError("Invalid bin number specified")
    isospearmandict.update({fn:{}})
    isospearmandict[fn].update({
        "qdict":        cp.deepcopy(qdict),
        "vdict":        cp.deepcopy(vdict),
        "varsysdict":   cp.deepcopy(varsysdict),
        "flatplots":    cp.deepcopy(flatplots),
        "distplots":    cp.deepcopy(distplots),
        })


pickle.dump(isospearmandict,open(f"{path}/{dictname}.p",'wb'))