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
from analib import Hist as HistL, PhysObj, Event, inc, fstrip, InputConfig, dphi, Hist2d as Hist2D
from analib import dframe as DataFrame
import pickle
import copy as cp
#from uproot_methods import TLorentzVector, TLorentzVectorArray
#from sklearn.model_selection import train_test_split
#from tensorflow.python.keras import backend as BE
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform as squareform

from ROOT import TFile, TH1F
import pdb

import mplhep as hep
# plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)
pd.set_option('mode.chained_assignment', None)

#evtlist = [35899001,24910172,106249475,126514437,43203653,27186346,17599588,64962950,61283040,54831588]

class Hist(HistL):
    def toTH1(s,title,scale=1,zerofill=1):
        # pdb.set_trace()
        if zerofill: 
            s[0][s[0]<=0] = zerofill
            s.ser[s.ser<=0] = zerofill*zerofill
        th1 = TH1F(title,title,len(s.hs[0]),s.hs[1][0],s.hs[1][-1])
        # th1 = TH1F(title,title,5,s.hs[1][0],s.hs[1][-1])
        for i in range(len(s.hs[0])):
            th1.SetBinContent(i+1,s.hs[0][i]*scale)
            th1.SetBinError(i+1,np.sqrt(s.ser[i])*scale)
            # th1.SetBinContent(i+1,s.hs[0][(i*5):((i+1)*5)].sum()*scale)
            # th1.SetBinError(i+1,np.sqrt(s.ser[(i*5):(i+1*5)].sum())*scale)
        pass
        return th1

    def errtoTH1(s,scale=1):#,zerofill=.00001):
        # if zerofill: s.ser[s.ser==0] = zerofill
        return np.histogram(s.hs[1][12:-5],s.hs[1][12:-4],weights=np.sqrt(s.ser[12:-4])*scale)


#%%

# useF = True
nbin = 17
RATIO = False
BLIND = False
path="Diststore"
refpath="Diststore"
folders = [f"{path}/Full/",f"{path}/QCDEnr/",f"{path}/QCDGen/",f"{path}/QCDInc/",
           f"{path}/TTbar/",f"{path}/WJets/",f"{path}/ZJets/",f"{path}/Data/"]
bgslice = ['Combined QCD','bEnriched' 'bGen' 'bInc' 'TTbar' 'WJets' 'ZJets','DATA']
namelist = ['Full','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
sysvars = ['pt','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','submass1','submass2','nsv','subtau1','subtau2','n2b1','Trig']
nit = 0
outdict = {}
# folders = ["Diststore/QCDGen/"]
## Create ROOT file for Combine
# os.remove(f"SubNet_CombinedX{nbin}.root")
rfile = TFile(f"SubNet_CombinedUNB{nbin}.root",'RECREATE')
th1b, quantsys, th1s = [], [], []
## controls weighting amount for shape uncertainties
wval = 1.1


Adat = pickle.load(open(f"{refpath}/Full/A.p",'rb'))
Bdat = pickle.load(open(f"{refpath}/Full/B.p",'rb'))
Cdat = pickle.load(open(f"{refpath}/Full/C.p",'rb'))
Fdat = pickle.load(open(f"{refpath}/Full/F.p",'rb'))
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

for f in folders:
    sigframe, bgframe = DataFrame(), DataFrame()
    sigframe['A'] = pickle.load(open(f"{f}Astt.p",'rb'))[:,0]
    sigframe['B'] = pickle.load(open(f"{f}Bstt.p",'rb'))
    sigframe['C'] = pickle.load(open(f"{f}Cstt.p",'rb'))[:,0]
    bgframe['A'] = pickle.load(open(f"{f}Abtt.p",'rb'))[:,0]
    bgframe['B'] = pickle.load(open(f"{f}Bbtt.p",'rb'))
    bgframe['C'] = pickle.load(open(f"{f}Cbtt.p",'rb'))[:,0]

    sigframe['F'] = pickle.load(open(f"{f}Fstt.p",'rb'))[:,0]
    bgframe['F'] = pickle.load(open(f"{f}Fbtt.p",'rb'))[:,0]

    Fdat = pickle.load(open(f"{f}F.p",'rb'))       

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
    fs = f.split('/')[1]
    distplots = {
        "sFA": Hist2D([FSensS,ASensB],None,"Full Network Signal","Subnet A signal",f"ratio/{fs}_FAsubdistS"),
        "sFB": Hist2D([FSensS,BSensB],None,"Full Network Signal","Subnet B signal",f"ratio/{fs}_FBsubdistS"),
        "sFC": Hist2D([FSensS,CSensB],None,"Full Network Signal","Subnet C signal",f"ratio/{fs}_FCsubdistS"),
        "bFA": Hist2D([FSensS,ASensB],None,"Full Network BG","Subnet A BG",f"ratio/{fs}_FAsubdistB"),
        "bFB": Hist2D([FSensS,BSensB],None,"Full Network BG","Subnet B BG",f"ratio/{fs}_FBsubdistB"),
        "bFC": Hist2D([FSensS,CSensB],None,"Full Network BG","Subnet C BG",f"ratio/{fs}_FCsubdistB"),
        "bAB": Hist2D([ASensB,BSensB],None,"Subnet A BG","Subnet B BG",f"ratio/{fs}_ABsubdistB"),
        "bAC": Hist2D([ASensB,CSensB],None,"Subnet A BG","Subnet C BG",f"ratio/{fs}_ACsubdistB"),
        "bBC": Hist2D([BSensB,CSensB],None,"Subnet B BG","Subnet C BG",f"ratio/{fs}_BCsubdistB"),
        }
    flatplots = {
        "sFA": Hist(abbounds,None,"20%-quantile A occupancy","Signal Events","FAflatS",f"ratio/{fs}_FA_signal"),
        "sFB": Hist(bbbounds,None,"20%-quantile B occupancy","Signal Events","FBflatS",f"ratio/{fs}_FB_signal"),
        "sFC": Hist(cbbounds,None,"20%-quantile C occupancy","Signal Events","FCflatS",f"ratio/{fs}_FC_signal"),
        "sAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Signal Events",f"ratio/{fs}_ABflatS","BinA"),
        "sAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Signal Events",f"ratio/{fs}_ACflatS","CinA"),
        "sBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Signal Events",f"ratio/{fs}_BCflatS","CinB"),
        "bFA": Hist(abbounds,None,"20%-quantile A occupancy","Background Events",f"ratio/{fs}_FAflatB","NetA"),
        "bFB": Hist(bbbounds,None,"20%-quantile B occupancy","Background Events",f"ratio/{fs}_FBflatB","NetB"),
        "bFC": Hist(cbbounds,None,"20%-quantile C occupancy","Background Events",f"ratio/{fs}_FCflatB","NetC"),
        "bAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Background Events",f"ratio/{fs}_ABflatB","BinA"),
        "bAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Background Events",f"ratio/{fs}_ACflatB","CinA"),
        "bBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Background Events",f"ratio/{fs}_BCflatB","CinB"),
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
        
    ## Is this blinding code?
    if BLIND:
        bgframe = bgframe[~((bgframe["Abin"] >= 3) & (bgframe["Bbin"] >= 3))]
        bgframe = bgframe[~((bgframe["Abin"] >= 3) & (bgframe["Cbin"] >= 3))]
        bgframe = bgframe[~((bgframe["Bbin"] >= 3) & (bgframe["Cbin"] >= 3))]
        sigframe = sigframe[~((sigframe["Abin"] >= 3) & (sigframe["Bbin"] >= 3))]
        sigframe = sigframe[~((sigframe["Abin"] >= 3) & (sigframe["Cbin"] >= 3))]
        sigframe = sigframe[~((sigframe["Bbin"] >= 3) & (sigframe["Cbin"] >= 3))]

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

    if 'Full' in f:
        if path != refpath:
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
        if 'Data' in f:
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
        flatplots["bFA"].fill(bgframe["A"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        flatplots["bFB"].fill(bgframe["B"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        flatplots["bFC"].fill(bgframe["C"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
        flatplots["bAB"].fill(bgframe["B"][bgframe["Abin"] == i] + i, bgframe["W"][bgframe["Abin"] == i])
        flatplots["bAC"].fill(bgframe["C"][bgframe["Abin"] == i] + i, bgframe["W"][bgframe["Abin"] == i])
        flatplots["bBC"].fill(bgframe["C"][bgframe["Bbin"] == i] + i, bgframe["W"][bgframe["Bbin"] == i])

#%%
    if nbin == 16:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
                # pass
            if not ('Data' in f and 's' in p):
                flatplots[p].ndivide(3)
            flatplots[p].plot(htype='bar',logv=False,error=True)
            for d in ['U','D']:
                for n in net:
                    for b in qdict[n]:
                        if "F" in p:
                            for i in range(16,25):
                                qdict[n][b][p][d][0][15] += qdict[n][b][p][d][0][i]
                                qdict[n][b][p][d].ser[15] += qdict[n][b][p][d].ser[i]
                            qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[16,17,18,19,20,21,22,23,24])
                            qdict[n][b][p][d].ser = np.delete(qdict[n][b][p][d].ser,[16,17,18,19,20,21,22,23,24])
                            qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-9,-8,-7,-6,-5,-4,-3,-2,-1])

                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                                # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                                #     qdict[n][b][p][d][0][i] = 0
                                #     flatplots[p][0][i] = 0
                        #     if BLIND:
                        #         qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19])
                        #         qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19])
                        #         qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-2,-1])
                        else: 
                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                            #     if ('Data' in f) and BLIND and i in [15,16,17,18,19,20,21,22,23,24]:
                            #         qdict[n][b][p][d][0][i] = 0
                            #         flatplots[p][0][i] = 0
                            # if BLIND:
                            #     qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19,23,24])
                            #     qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19,23,24])
                            #     qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-4,-3,-2,-1])
                        qdict[n][b][p][d].ndivide(3)
            if "F" in p:
                for i in range(16,25):
                    flatplots[p][0][15] += flatplots[p][0][i]
                    flatplots[p].ser[15] += flatplots[p].ser[i]
                flatplots[p][0] = np.delete(flatplots[p][0],[16,17,18,19,20,21,22,23,24])
                flatplots[p].ser = np.delete(flatplots[p].ser,[16,17,18,19,20,21,22,23,24])
                flatplots[p][1] = np.delete(flatplots[p][1],[-9,-8,-7,-6,-5,-4,-3,-2,-1])
                # if BLIND:
                #     flatplots[p][0] = np.delete(flatplots[p][0],[18,19])
                #     flatplots[p].ser = np.delete(flatplots[p].ser,[18,19])
                #     flatplots[p][1] = np.delete(flatplots[p][1],[-2,-1])
            # elif BLIND: 
            #     flatplots[p][0] = np.delete(flatplots[p][0],[18,19,23,24])
            #     flatplots[p].ser = np.delete(flatplots[p].ser,[18,19,23,24])
            #     flatplots[p][1] = np.delete(flatplots[p][1],[-4,-3,-2,-1])
        for d in ['U','D']:
            for p in ["bFA3D","bFB3D","bFC3D"]:
                for n in net:
                    for b in qdict[n]:
                        for i in range(16,25):
                            qdict[n][b][p][d][0][15] += qdict[n][b][p][d][0][i]
                            qdict[n][b][p][d].ser[15] += qdict[n][b][p][d].ser[i]

                        qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[16,17,18,19,20,21,22,23,24])
                        qdict[n][b][p][d].ser = np.delete(qdict[n][b][p][d].ser,[16,17,18,19,20,21,22,23,24])
                        qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-9,-8,-7,-6,-5,-4,-3,-2,-1])
                        # for i in range(15,20):
                        #     qdict[n][b][p][d][0][i] += qdict[n][b][p][d][0][i+5]
                        #     qdict[n][b][p][d].ser[i] += qdict[n][b][p][d].ser[i+5]
                        # qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[20,21,22,23,24])
                        # qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[20,21,22,23,24])
                        # qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-5,-4,-3,-2,-1])
                        for i in range(qdict[n][b][p][d][1].shape[0]):
                            qdict[n][b][p][d][1][i] = 0.2 * i
                            # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                            #     qdict[n][b][p][d][0][i] = 0
                            # pass
                        qdict[n][b][p][d].ndivide(3)
                        # if BLIND:
                        #     qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19,23,24])
                        #     qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19,23,24])
                        #     qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-4,-3,-2,-1])
            for n in vdict:
                for s in vdict[n]:
                    for l in vdict[n][s]:
                        for b in vdict[n][s][l]:
                            vdict[n][s][l][b][d].ndivide(3)
                            if "F" in n:
                                for i in range(16,25):
                                    vdict[n][s][l][b][d][0][15] += vdict[n][s][l][b][d][0][i]
                                    vdict[n][s][l][b][d].ser[15] += vdict[n][s][l][b][d].ser[i]

                                vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[16,17,18,19,20,21,22,23,24])
                                vdict[n][s][l][b][d].ser = np.delete(vdict[n][s][l][b][d].ser,[16,17,18,19,20,21,22,23,24])
                                vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-9,-8,-7,-6,-5,-4,-3,-2,-1])
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
                                    # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                                    #     vdict[n][s][l][b][d][0][i] = 0
                                # if BLIND:
                                #     vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[18,19])
                                #     vdict[n][s][l][b][d].ser= np.delete(vdict[n][s][l][b][d].ser,[18,19])
                                #     vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-2,-1])
                                
                            else:
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
                                #     if ('Data' in f) and BLIND and i in [15,16,17,18,19,20,21,22,23,24]:
                                #         vdict[n][s][l][b][d][0][i] = 0
                                # if BLIND:
                                #     vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[18,19,23,24])
                                #     vdict[n][s][l][b][d].ser= np.delete(vdict[n][s][l][b][d].ser,[18,19,23,24])
                                #     vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-4,-3,-2,-1])
                                
    elif nbin == 17:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
            if not ('Data' in f and 's' in p):
                flatplots[p].ndivide(3)
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

    
    elif nbin == 20:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
                # pass
            if not ('Data' in f and 's' in p):
                flatplots[p].ndivide(3)
            flatplots[p].plot(htype='bar',logv=False,error=True)
            for d in ['U','D']:
                for n in net:
                    for b in qdict[n]:
                        if "F" in p:                           
                            for i in range(15,20):
                                qdict[n][b][p][d][0][i] += qdict[n][b][p][d][0][i+5]
                                qdict[n][b][p][d].ser[i] += qdict[n][b][p][d].ser[i+5]         
                            qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[20,21,22,23,24])
                            qdict[n][b][p][d].ser = np.delete(qdict[n][b][p][d].ser,[20,21,22,23,24])
                            qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-5,-4,-3,-2,-1])
                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                                # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                                #     qdict[n][b][p][d][0][i] = 0
                                #     flatplots[p][0][i] = 0
                        #     if BLIND:
                        #         qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19])
                        #         qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19])
                        #         qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-2,-1])
                        else: 
                            for i in range(qdict[n][b][p][d][1].shape[0]):
                                qdict[n][b][p][d][1][i] = 0.2 * i
                                # if ('Data' in f) and BLIND and i in [15,16,17,18,19,20,21,22,23,24]:
                                #     qdict[n][b][p][d][0][i] = 0
                                #     flatplots[p][0][i] = 0
                            # if BLIND:
                            #     qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19,23,24])
                            #     qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19,23,24])
                            #     qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-4,-3,-2,-1])
                        qdict[n][b][p][d].ndivide(3)
            if "F" in p:
                for i in range(15,20):
                    flatplots[p][0][i] += flatplots[p][0][i+5]
                    flatplots[p].ser[i] += flatplots[p].ser[i+5]
                flatplots[p][0] = np.delete(flatplots[p][0],[20,21,22,23,24])
                flatplots[p].ser = np.delete(flatplots[p].ser,[20,21,22,23,24])
                flatplots[p][1] = np.delete(flatplots[p][1],[-5,-4,-3,-2,-1])
                # if BLIND:
                #     flatplots[p][0] = np.delete(flatplots[p][0],[18,19])
                #     flatplots[p].ser = np.delete(flatplots[p].ser,[18,19])
                #     flatplots[p][1] = np.delete(flatplots[p][1],[-2,-1])
            # elif BLIND: 
            #     flatplots[p][0] = np.delete(flatplots[p][0],[18,19,23,24])
            #     flatplots[p].ser = np.delete(flatplots[p].ser,[18,19,23,24])
            #     flatplots[p][1] = np.delete(flatplots[p][1],[-4,-3,-2,-1])
        for d in ['U','D']:
            for p in ["bFA3D","bFB3D","bFC3D"]:
                for n in net:
                    for b in qdict[n]:
                        for i in range(15,20):
                            qdict[n][b][p][d][0][i] += qdict[n][b][p][d][0][i+5]
                            qdict[n][b][p][d].ser[i] += qdict[n][b][p][d].ser[i+5]
                        qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[20,21,22,23,24])
                        qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[20,21,22,23,24])
                        qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-5,-4,-3,-2,-1])
                        for i in range(qdict[n][b][p][d][1].shape[0]):
                            qdict[n][b][p][d][1][i] = 0.2 * i
                            # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                            #     qdict[n][b][p][d][0][i] = 0
                            # pass
                        qdict[n][b][p][d].ndivide(3)
                        # if BLIND:
                        #     qdict[n][b][p][d][0] = np.delete(qdict[n][b][p][d][0],[18,19,23,24])
                        #     qdict[n][b][p][d].ser= np.delete(qdict[n][b][p][d].ser,[18,19,23,24])
                        #     qdict[n][b][p][d][1] = np.delete(qdict[n][b][p][d][1],[-4,-3,-2,-1])
            for n in vdict:
                for s in vdict[n]:
                    for l in vdict[n][s]:
                        for b in vdict[n][s][l]:
                            vdict[n][s][l][b][d].ndivide(3)
                            if "F" in n:
                                for i in range(15,20):
                                    vdict[n][s][l][b][d][0][i] += vdict[n][s][l][b][d][0][i+5]
                                    vdict[n][s][l][b][d].ser[i] += vdict[n][s][l][b][d].ser[i+5]
                                vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[20,21,22,23,24])
                                vdict[n][s][l][b][d].ser = np.delete(vdict[n][s][l][b][d].ser,[20,21,22,23,24])
                                vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-5,-4,-3,-2,-1])
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
                                    # if ('Data' in f) and BLIND and i in [15,16,17,18,19]:
                                    #     vdict[n][s][l][b][d][0][i] = 0
                                # if BLIND:
                                #     vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[18,19])
                                #     vdict[n][s][l][b][d].ser= np.delete(vdict[n][s][l][b][d].ser,[18,19])
                                #     vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-2,-1])
                                
                            else:
                                for i in range(vdict[n][s][l][b][d][1].shape[0]):
                                    vdict[n][s][l][b][d][1][i] = 0.2 * i
                                    # if ('Data' in f) and BLIND and i in [15,16,17,18,19,20,21,22,23,24]:
                                    #     vdict[n][s][l][b][d][0][i] = 0
                                # if BLIND:
                                #     vdict[n][s][l][b][d][0] = np.delete(vdict[n][s][l][b][d][0],[18,19,23,24])
                                #     vdict[n][s][l][b][d].ser= np.delete(vdict[n][s][l][b][d].ser,[18,19,23,24])
                                #     vdict[n][s][l][b][d][1] = np.delete(vdict[n][s][l][b][d][1],[-4,-3,-2,-1])
    elif nbin == 25:
        for p in flatplots:
            for i in range(flatplots[p][1].shape[0]):
                flatplots[p][1][i] = 0.2 * i
                # pass
            if not ('Data' in f and 's' in p):
                flatplots[p].ndivide(3)
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
                    
    else: raise ValueError("Invalid bin number specified")
    #%%
    if not 'Data' in f:
        bslice = f.split('/')[-2]
        colors = ['red','orange','gold','green','skyblue','mediumpurple','plum']
        leg = ['bin 0','bin 1','bin 2','bin 3','bin 4','standard']
        plt.clf()
        for net in ["A","B","C"]:
            for subn in ["bFA","bFB","bFC"] + ["bAB","bAC","bBC"]:
                for b in range(5):
                    # quantsys.append(qdict[net][b][subn]['U'].toTH1(bslice + "_" + qdict[net][b][subn]['U'].title + f"_Qsys{net}{b}_{bslice}Up"))
                    # quantsys.append(qdict[net][b][subn]['D'].toTH1(bslice + "_" + qdict[net][b][subn]['D'].title + f"_Qsys{net}{b}_{bslice}Down"))
                    quantsys.append(qdict[net][b][subn]['U'].toTH1(bslice + "_" + qdict[net][b][subn]['U'].title + f"_Qsys{net}{b}Up"))
                    quantsys.append(qdict[net][b][subn]['D'].toTH1(bslice + "_" + qdict[net][b][subn]['D'].title + f"_Qsys{net}{b}Down"))
        for subn in ["bFA3D", "bFB3D", "bFC3D"]:
            # quantsys.append(qdict[net][0][subn]['U'].toTH1(f"{bslice}_{qdict[net][0][subn]['U'].title}_{bslice}Up"))
            # quantsys.append(qdict[net][0][subn]['D'].toTH1(f"{bslice}_{qdict[net][0][subn]['D'].title}_{bslice}Down"))
            quantsys.append(qdict[net][0][subn]['U'].toTH1(f"{bslice}_{qdict[net][0][subn]['U'].title}Up"))
            quantsys.append(qdict[net][0][subn]['D'].toTH1(f"{bslice}_{qdict[net][0][subn]['D'].title}Down"))
        plt.clf()
        for n in ["AB","AC","BC","FA","FB","FC"]:
            th1b.append(flatplots[f"b{n}"].toTH1(bslice+"_"+flatplots[f"b{n}"].title+""))
            distplots[f"b{n}"].plot(text=True,fontsize=14,vmin=0,vmax=2)
            for sub in ["AB","AC","BC"]:
                for l in sub:
                    for b in range(5):
                        # quantsys.append(vdict[n][sub][l][b]['U'].toTH1(f"{bslice}_{vdict[n][sub][l][b]['U'].title}_VQsys{sub}{b}{l}_{bslice}Up"))
                        # quantsys.append(vdict[n][sub][l][b]['D'].toTH1(f"{bslice}_{vdict[n][sub][l][b]['D'].title}_VQsys{sub}{b}{l}_{bslice}Down"))
                        quantsys.append(vdict[n][sub][l][b]['U'].toTH1(f"{bslice}_{vdict[n][sub][l][b]['U'].title}_VQsys{sub}{b}{l}Up"))
                        quantsys.append(vdict[n][sub][l][b]['D'].toTH1(f"{bslice}_{vdict[n][sub][l][b]['D'].title}_VQsys{sub}{b}{l}Down"))
        for n in ["AB","AC","BC"]:
            # quantsys.append(vdict[n][sub][l]["3D"]['U'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_F3D_{bslice}Up"))
            # quantsys.append(vdict[n][sub][l]["3D"]['D'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_F3D_{bslice}Down"))
            quantsys.append(vdict[n][sub][l]["3D"]['U'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_F3DUp"))
            quantsys.append(vdict[n][sub][l]["3D"]['D'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_F3DDown"))
        for n in ["AB","AC","BC","FA","FB","FC"]:
            plt.clf()

            for sub in ["AB","AC","BC"]:
                    if "F" in n: tstr = f"Net{n[-1]}"
                    else: tstr = f"{n[-1]}in{n[0]}"
                    # quantsys.append(vdict[n][sub][sub[0]]["F"]['U'].toTH1(f"{bslice}_{tstr}_{sub}2D_{bslice}Up"))
                    # quantsys.append(vdict[n][sub][sub[0]]["F"]['D'].toTH1(f"{bslice}_{tstr}_{sub}2D_{bslice}Down"))
                    quantsys.append(vdict[n][sub][sub[0]]["F"]['U'].toTH1(f"{bslice}_{tstr}_{sub}2DUp"))
                    quantsys.append(vdict[n][sub][sub[0]]["F"]['D'].toTH1(f"{bslice}_{tstr}_{sub}2DDown"))
    # if 'QCDInc' in f:
    #     qslice = 'QCDIncExtra'
    #     colors = ['red','orange','gold','green','skyblue','mediumpurple','plum']
    #     leg = ['bin 0','bin 1','bin 2','bin 3','bin 4','standard']
    #     plt.clf()
    #     for net in ["A","B","C"]:
    #         # th1b.append(flatplots[f"bF{net}"].toTH1(bslice+"_"+flatplots[f"bF{net}"].title))
    #         for subn in ["bFA","bFB","bFC"] + ["bAB","bAC","bBC"]:
    #             for b in range(5):
    #                 # quantsys.append(qdict[net][b][subn]['U'].nmult(3)
    #                 quantsys.append(qdict[net][b][subn]['U'].toTH1(qslice + "_" + qdict[net][b][subn]['U'].title + f"_Qsys{net}{b}Up"))
    #                 quantsys.append(qdict[net][b][subn]['D'].toTH1(qslice + "_" + qdict[net][b][subn]['D'].title + f"_Qsys{net}{b}Down"))
    #                 # qdict[net][b][subn].make(htype='step',color=colors[b])
    #                 # tempdown.make(htype='step',color=colors[b])
    #             # tempratio = cp.deepcopy(flatplots[subn[:3]])
    #             # tempratio.title = f"Changed {net}"
    #             # tempratio.fname = 'ratio/' + bslice + "_" + flatplots[subn[:3]].title + f"_Qsys{net}D"
    #             # tempratio.plot(same=True,htype='step',legend=leg,color='black')
    #     for subn in ["bFA3D", "bFB3D", "bFC3D"]:
    #         quantsys.append(qdict[net][0][subn]['U'].toTH1(f"{qslice}_{qdict[net][0][subn]['U'].title}Up"))
    #         quantsys.append(qdict[net][0][subn]['D'].toTH1(f"{qslice}_{qdict[net][0][subn]['D'].title}Down"))
    #         # qdict[net][b][subn].make(htype='step',color=colors[b])
    #         # tempdown.make(htype='step',color=colors[b])
    #         # tempratio = cp.deepcopy(flatplots[subn[:3]])
    #         # tempratio.title = f"Changed {net}"
    #         # tempratio.fname = 'ratio/' + bslice + "_" + flatplots[subn[:3]].title + "D"
    #         # tempratio.plot(same=True,htype='step',legend=leg,color='black')
    #     plt.clf()
    #     for n in ["AB","AC","BC","FA","FB","FC"]:
    #         # flatplots[f"b{n}"][0][flatplots[f"b{n}"][0] == 0] = 0.01
    #         th1b.append(flatplots[f"b{n}"].toTH1(qslice+"_"+flatplots[f"b{n}"].title))
    #         for sub in ["AB","AC","BC"]:
    #             for l in sub:
    #                 # vdict[n][sub][l]["F"].fname = f"ratio/{bslice}_plot{n}_weight{sub}_q{l}FUp"
    #                 # vdict[n][sub][l]["F"].plot(htype='step',color='black')
    #                 # vdict[n][sub][l]["3D"].fname = f"ratio/{bslice}_plot{n}_3DDown"
    #                 # vdict[n][sub][l]["3D"].plot(htype='step',color='black')
    #                 for b in range(5):
    #                     # tempdown = cp.deepcopy(flatplots[f"b{n}"])
    #                     quantsys.append(vdict[n][sub][l][b]['U'].toTH1(f"{qslice}_{vdict[n][sub][l][b]['U'].title}_VQsys{sub}{b}{l}Up"))
    #                     # vdict[n][sub][l][b].make(htype='step',color=colors[b])
    #                     # tempdown[0] = tempdown[0] * tempdown[0] / (vdict[n][sub][l][b][0] + 1E-9)
    #                     # tempdown[0][~(np.isfinite(tempdown[0]))] = 0
    #                     quantsys.append(vdict[n][sub][l][b]['D'].toTH1(f"{qslice}_{vdict[n][sub][l][b]['D'].title}_VQsys{sub}{b}{l}Down"))
    #                     # tempdown.make(htype='step',color=colors[b])
    #                 # temphist = cp.deepcopy(flatplots[f"b{n}"])
    #                 # temphist.title = f"From {n} with {sub} weights in {l} quantiles"
    #                 # if "F" in n: temphist.fname = f"ratio/{bslice}_Net{n[-1]}_weight{sub}_q{l}D"
    #                 # else: temphist.fname = f"ratio/{bslice}_{n[-1]}in{n[0]}_weight{sub}_q{l}D"
    #                 # temphist.plot(same=True,htype='step',legend=leg,color='black')
    
    #     for n in ["AB","AC","BC"]:
    #         # for sub in ["AB","AC","BC"]:
    #             # l = sub[0]
    #             # vdict[n][sub][l]["F"].fname = f"ratio/{bslice}_{n[-1]}in{n[0]}_weight{sub}_qFUp"
    #             # vdict[n][sub][l]["F"].plot(htype='step',color='black')
    #             # quantsys.append(vdict[n][sub][l]["F"]['U'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_weight{sub}_qFUp"))
    #             # tempdown = cp.deepcopy(flatplots[f"b{n}"])
    #             # tempdown[0] = tempdown[0] * tempdown [0] / (vdict[n][sub][l]["F"][0] + 1E-9)
    #             # tempdown[0][~(tempdown[0] > 0)] = 0
    #             # tempdown.fname = f"ratio/{bslice}_plot{n}_weight{sub}_qFDown"
    #             # quantsys.append(vdict[n][sub][l]["F"]['D'].toTH1(f"{bslice}_{n[-1]}in{n[0]}_weight{sub}_qFDown"))
    #             # tempdown.plot(htype='step',color='black')
    #         # vdict[n][sub][l]["3D"].fname = f"ratio/{bslice}_plot{n}_F3DDown"
    #         # vdict[n][sub][l]["3D"].plot(htype='step',color='black')
    #         quantsys.append(vdict[n][sub][l]["3D"]['U'].toTH1(f"{qslice}_{n[-1]}in{n[0]}_F3DUp"))
    #         # tempdown = cp.deepcopy(flatplots[f"b{n}"])
    #         # tempdown[0] = tempdown[0] * tempdown [0] / (vdict[n][sub][l]["3D"][0] + 1E-9)
    #         # tempdown[0][~(tempdown[0] > 0)] = 0
    #         # tempdown.fname = f"ratio/{bslice}_plot{n}_F3DUp"
    #         quantsys.append(vdict[n][sub][l]["3D"]['D'].toTH1(f"{qslice}_{n[-1]}in{n[0]}_F3DDown"))
    #         # tempdown.plot(htype='step',color='black')
    #     for n in ["AB","AC","BC","FA","FB","FC"]:
    #         plt.clf()
    #         # cint = 0
    #         for sub in ["AB","AC","BC"]:
    #                 # tempdown = cp.deepcopy(flatplots[f"b{n}"])
    #                 # vdict[n][sub][sub[0]]["F"].make(htype='step',color=colors[cint])
    #                 if "F" in n: tstr = f"Net{n[-1]}"
    #                 else: tstr = f"{n[-1]}in{n[0]}"
    #                 quantsys.append(vdict[n][sub][sub[0]]["F"]['U'].toTH1(f"{qslice}_{tstr}_{sub}2DUp"))
    #                 # tempdown[0] = tempdown[0] * tempdown [0] / (vdict[n][sub][sub[0]]["F"][0] + 1E-9)
    #                 # tempdown[0][~(tempdown[0] > 0)] = 0
    #                 quantsys.append(vdict[n][sub][sub[0]]["F"]['D'].toTH1(f"{qslice}_{tstr}_{sub}2DDown"))
    #                 # tempdown.make(htype='step',color=colors[cint])
    #                 # cint += 1
    #         # flatplots[f"b{n}"].fname = f"ratio/{bslice}_plot{n}_2DD"
    #         # flatplots[f"b{n}"].plot(htype='step',color='k',legend=["AB","AC","BC","Flat"],same=True)
    outdict.update({bslice:flatplots})
    outdict.update({bslice+'2d':distplots})
    outdict.update({bslice+'v':vdict})
    outdict.update({bslice+'q':qdict})
    nit += 1
    if 'Data' in f:
        for net in ["A","B","C"]:
            th1s.append(flatplots[f"sF{net}"].toTH1("data_obs_Net"+net,zerofill=False))
        for n in ["AB","AC","BC"]:
            th1s.append(flatplots[f"s{n}"].toTH1(f"data_obs_{n[-1]}in{n[0]}",zerofill=False))
    elif 'Full' in f:
        for net in ["A","B","C"]:
            th1s.append(flatplots[f"sF{net}"].toTH1("SignalMC_Net"+net,zerofill=.001))
        for nets in ["AB","AC","BC"]:
            th1s.append(flatplots[f"s{nets}"].toTH1(f"SignalMC_{nets[1]}in{nets[0]}",zerofill=.001))

    
rfile.Write()
for elem in th1s + th1b:
    elem.SetDirectory(0)
rfile.Close()
pickle.dump(outdict,open(f"Snetplots/{path}_{nbin}.p",'wb'))