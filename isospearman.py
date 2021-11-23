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
    def toTH1(s,title,scale=1):
        # pdb.set_trace()
        th1 = TH1F(title,title,len(s.hs[0]),s.hs[1][0],s.hs[1][-1])
        for i in range(len(s.hs[0])):
            th1.SetBinContent(i+1,s.hs[0][i]*scale)
            th1.SetBinError(i+1,np.sqrt(s.ser[i])*scale)
        pass
        return th1

    def errtoTH1(s,scale=1):
        return np.histogram(s.hs[1][12:-5],s.hs[1][12:-4],weights=np.sqrt(s.ser[12:-4])*scale)


#%%

# useF = True

RATIO = False
folders = ["Diststore/Full/","Diststore/QCDEnr/","Diststore/QCDGen/","Diststore/QCDInc/",
           "Diststore/TTbar/","Diststore/WJets/","Diststore/ZJets/",'Diststore/Data/']
namelist = ['Full','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
nit = 0
outdict = {}
# folders = ["Diststore/QCDGen/"]
## Create ROOT file for Combine
os.remove('SubNet_Combined.root')
rfile = TFile('SubNet_Combined.root','UPDATE')
th1b, quantsys = [], []
# folders = ['Diststore/Data/']

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


    Adat = pickle.load(open(f"{f}A.p",'rb'))
    Bdat = pickle.load(open(f"{f}B.p",'rb'))
    Cdat = pickle.load(open(f"{f}C.p",'rb'))
    Fdat = pickle.load(open(f"{f}F.p",'rb'))

    for elem in [Adat, Bdat, Cdat, Fdat]:
        elem["SensB"] = [0] + list(elem["SensB"]) + [1]
        elem["SensS"] = list(elem["SensS"])

        for i in range(1,6):
            elem["SensB"].pop(i)
            elem["SensS"].pop(i)

    # sbounds = Fdat["SensS"] + [x + 1 for x in Fdat["SensS"][1:]] + [x + 2 for x in Fdat["SensS"][1:]] +\
    #         [x + 3 for x in Fdat["SensS"][1:]] + [x + 4 for x in Fdat["SensS"][1:]]
    asbounds = Adat["SensS"] + [x + 1 for x in Adat["SensS"][1:]] + [x + 2 for x in Adat["SensS"][1:]] +\
            [x + 3 for x in Adat["SensS"][1:]] + [x + 4 for x in Adat["SensS"][1:]]
    bsbounds = Bdat["SensS"] + [x + 1 for x in Bdat["SensS"][1:]] + [x + 2 for x in Bdat["SensS"][1:]] +\
            [x + 3 for x in Bdat["SensS"][1:]] + [x + 4 for x in Bdat["SensS"][1:]]
    csbounds = Cdat["SensS"] + [x + 1 for x in Cdat["SensS"][1:]] + [x + 2 for x in Cdat["SensS"][1:]] +\
            [x + 3 for x in Cdat["SensS"][1:]] + [x + 4 for x in Cdat["SensS"][1:]]
    # bbounds = Fdat["SensB"] + [x + 1 for x in Fdat["SensB"][1:]] + [x + 2 for x in Fdat["SensB"][1:]] +\
    #         [x + 3 for x in Fdat["SensB"][1:]] + [x + 4 for x in Fdat["SensB"][1:]]
    abbounds = Adat["SensB"] + [x + 1 for x in Adat["SensB"][1:]] + [x + 2 for x in Adat["SensB"][1:]] +\
            [x + 3 for x in Adat["SensB"][1:]] + [x + 4 for x in Adat["SensB"][1:]]
    bbbounds = Bdat["SensB"] + [x + 1 for x in Bdat["SensB"][1:]] + [x + 2 for x in Bdat["SensB"][1:]] +\
            [x + 3 for x in Bdat["SensB"][1:]] + [x + 4 for x in Bdat["SensB"][1:]]
    cbbounds = Cdat["SensB"] + [x + 1 for x in Cdat["SensB"][1:]] + [x + 2 for x in Cdat["SensB"][1:]] +\
            [x + 3 for x in Cdat["SensB"][1:]] + [x + 4 for x in Cdat["SensB"][1:]]
    distplots = {
        "sFA": Hist2D([Fdat['SensS'],Adat['SensB']],None,"Full Network Signal","Subnet A signal","FAsubdistS"),
        "sFB": Hist2D([Fdat['SensS'],Bdat['SensB']],None,"Full Network Signal","Subnet B signal","FBsubdistS"),
        "sFC": Hist2D([Fdat['SensS'],Cdat['SensB']],None,"Full Network Signal","Subnet C signal","FCsubdistS"),
        "bFA": Hist2D([Fdat['SensS'],Adat['SensB']],None,"Full Network BG","Subnet A BG","FAsubdistB"),
        "bFB": Hist2D([Fdat['SensS'],Bdat['SensB']],None,"Full Network BG","Subnet B BG","FBsubdistB"),
        "bFC": Hist2D([Fdat['SensS'],Cdat['SensB']],None,"Full Network BG","Subnet C BG","FCsubdistB"),
        "bAB": Hist2D([Adat['SensB'],Bdat['SensB']],None,"Subnet A BG","Subnet B BG","ABsubdistB"),
        "bAC": Hist2D([Adat['SensB'],Cdat['SensB']],None,"Subnet A BG","Subnet C BG","ACsubdistB"),
        "bBC": Hist2D([Bdat['SensB'],Cdat['SensB']],None,"Subnet B BG","Subnet C BG","BCsubdistB"),
        }
    flatplots = {
        "sFA": Hist(abbounds,None,"20%-quantile A occupancy","Signal Events","FAflatS","FA_signal"),
        "sFB": Hist(bbbounds,None,"20%-quantile B occupancy","Signal Events","FBflatS","FB_signal"),
        "sFC": Hist(cbbounds,None,"20%-quantile C occupancy","Signal Events","FCflatS","FC_signal"),
        "sAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Signal Events","ABflatS","BinA"),
        "sAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Signal Events","ACflatS","CinA"),
        "sBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Signal Events","BCflatS","CinB"),
        "bFA": Hist(abbounds,None,"20%-quantile A occupancy","Background Events","FAflatB","NetA"),
        "bFB": Hist(bbbounds,None,"20%-quantile B occupancy","Background Events","FBflatB","NetB"),
        "bFC": Hist(cbbounds,None,"20%-quantile C occupancy","Background Events","FCflatB","NetC"),
        "bAB": Hist(bbbounds,None,"20%-quantile B occupancy in A","Background Events","ABflatB","BinA"),
        "bAC": Hist(cbbounds,None,"20%-quantile C occupancy in A","Background Events","ACflatB","CinA"),
        "bBC": Hist(cbbounds,None,"20%-quantile C occupancy in B","Background Events","BCflatB","CinB"),
        }

    sigframe["W"] = Fdat["WS"].reset_index(drop=True)
    bgframe["W"] = Fdat["WB"].reset_index(drop=True)
    # sigframe["Abin"], sigframe["Bbin"], sigframe["Cbin"] = 5, 5, 5
    # bgframe["Abin"], bgframe["Bbin"], bgframe["Cbin"] = 5, 5, 5
    sigframe ["Fbin"], bgframe["Fbin"] = 5, 5
    bgframe["Abin"], bgframe["Bbin"], bgframe["Cbin"] = 5, 5, 5
    sigframe["Abin"], sigframe["Bbin"], sigframe["Cbin"] = 5, 5, 5
    ABCtensor = {}
    for i in range(5):
        ABCtensor.update({i:cp.deepcopy(distplots["bAB"])})
    for i in range(5,0,-1):
        sigframe["Abin"][sigframe["A"] <= Adat["SensB"][i]] = i-1
        sigframe["Bbin"][sigframe["B"] <= Bdat["SensB"][i]] = i-1
        sigframe["Cbin"][sigframe["C"] <= Cdat["SensB"][i]] = i-1
        sigframe["Fbin"][sigframe["F"] <= Fdat["SensS"][i]] = i-1
        bgframe["Abin"][bgframe["A"] <= Adat["SensB"][i]] = i-1
        bgframe["Bbin"][bgframe["B"] <= Bdat["SensB"][i]] = i-1
        bgframe["Cbin"][bgframe["C"] <= Cdat["SensB"][i]] = i-1
        bgframe["Fbin"][bgframe["F"] <= Fdat["SensS"][i]] = i-1

    distplots["sFA"].fill(sigframe["F"],sigframe["A"],Adat['WS'])
    distplots["sFB"].fill(sigframe["F"],sigframe["B"],Adat['WS'])
    distplots["sFC"].fill(sigframe["F"],sigframe["C"],Adat['WS'])
    distplots["bFA"].fill(bgframe["F"],bgframe["A"],Adat['WB'])
    distplots["bFB"].fill(bgframe["F"],bgframe["B"],Adat['WB'])
    distplots["bFC"].fill(bgframe["F"],bgframe["C"],Adat['WB'])
    distplots["bAB"].fill(bgframe["A"],bgframe["B"],Adat['WB'])
    distplots["bAC"].fill(bgframe["A"],bgframe["C"],Adat['WB'])
    distplots["bBC"].fill(bgframe["B"],bgframe["C"],Adat['WB'])

    for i in range(5):
        ABCtensor[i].fill(bgframe["A"][bgframe["Cbin"] == i],bgframe["B"][bgframe["Cbin"] == i],Adat['WB'][bgframe["Cbin"] == i])

    for p in distplots:
        distplots[p][1] = [0,1,2,3,4,5]
        distplots[p][2] = [0,1,2,3,4,5]
        distplots[p][0] /= (distplots[p][0].sum()/25)
        distplots[p].plot(text=True,edgecolor='k',tlen=5)

    bounds = [abbounds, bbbounds, cbbounds]
    qdict = {"A":{},"B":{},"C":{}}
    net = ["A","B","C"]
    debugdict = {}
    for n in range(3):
        for b in range(5):
            qdict[net[n]].update({b:{}})
            tempwB, tempwS = cp.deepcopy(bgframe["W"]), cp.deepcopy(sigframe["W"])
            tempwS[sigframe[f"{net[n]}bin"] == b] *= 1.5
            tempwB[bgframe[f"{net[n]}bin"] == b] *= 1.5
            for key in flatplots:
                qdict[net[n]][b].update({key:cp.deepcopy(flatplots[key])})
            for i in range(5):
                qdict[net[n]][b]["bAB"].fill(bgframe["B"][bgframe["Abin"] == i] + i, tempwB[bgframe["Abin"] == i])
                qdict[net[n]][b]["bAC"].fill(bgframe["C"][bgframe["Abin"] == i] + i, tempwB[bgframe["Abin"] == i])
                qdict[net[n]][b]["bBC"].fill(bgframe["C"][bgframe["Bbin"] == i] + i, tempwB[bgframe["Bbin"] == i])
                qdict[net[n]][b]["bFA"].fill(bgframe["A"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                qdict[net[n]][b]["bFB"].fill(bgframe["B"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                qdict[net[n]][b]["bFC"].fill(bgframe["C"][bgframe["Fbin"] == i] + i, tempwB[bgframe["Fbin"] == i])
                # qdict[net[n]][b]["sFA"].fill(sigframe["A"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                # qdict[net[n]][b]["sFB"].fill(sigframe["B"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                # qdict[net[n]][b]["sFC"].fill(sigframe["C"][bgframe["Fbin"] == i] + i, tempwS[bgframe["Fbin"] == i])
                # qdict[net[n]][b]["sAB"].fill(sigframe["B"][bgframe["Abin"] == i] + i, tempwS[bgframe["Abin"] == i])
                # qdict[net[n]][b]["sAC"].fill(sigframe["C"][bgframe["Abin"] == i] + i, tempwS[bgframe["Abin"] == i])
                # qdict[net[n]][b]["sBC"].fill(sigframe["C"][bgframe["Bbin"] == i] + i, tempwS[bgframe["Bbin"] == i])
    vdict = {}
    for n in ["AB","AC","BC"]:#,"FA","FB","FC"]:
        vdict.update({n:{}})
        for sub in ["AB","AC","BC"]:
            vdict[n].update({sub:{}})
            for l in sub:
                vdict[n][sub].update({l:{}})
                test = cp.deepcopy(bgframe["W"])
                for b in range(5):
                    vdict[n][sub][l].update({b:cp.deepcopy(flatplots[f"b{n}"])})
                    for i in range(5):
                        tempwB = cp.deepcopy(bgframe["W"])
                        for j in range(5):
                            if l == sub[0]:
                                tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] *= distplots[f"b{sub}"][0][b][j]
                                if i == 0:
                                    test[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[1]}bin"] == j)] *= distplots[f"b{sub}"][0][b][j]
                                debugdict.update({f"{n}{sub}{l}{b}{j}":distplots[f"b{sub}"][0][j][b]})
                            else:
                                tempwB[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[0]}bin"] == j)] *= distplots[f"b{sub}"][0][j][b]
                                # test[(bgframe[f"{l}bin"] == b) & (bgframe[f"{sub[0]}bin"] == j)] *= distplots[f"b{sub}"][0][j][b]
                                debugdict.update({f"{n}{sub}{l}{b}{j}":distplots[f"b{sub}"][0][j][b]})
                        vdict[n][sub][l][b].fill(bgframe[f"{n[-1]}"][bgframe[f"{n[0]}bin"] == i] + i, tempwB[bgframe[f"{n[0]}bin"] == i])
                vdict[n][sub][l].update({"F":cp.deepcopy(flatplots[f"b{n}"])})
                for i in range(5):
                    vdict[n][sub][l]["F"].fill(bgframe[f"{n[-1]}"][bgframe[f"{n[0]}bin"] == i] + i, test[bgframe[f"{n[0]}bin"] == i])

    for i in range(5):
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


    for p in flatplots:
        for i in range(26):
            flatplots[p][1][i] = 0.2 * i
            # pass
        flatplots[p].ndivide(3)
        flatplots[p].plot(htype='bar',logv=False,error=True)
        for n in net:
            for b in qdict[n]:
                for i in range(26):
                    qdict[n][b][p][1][i] = 0.2 * i
                    if ('Data' in f) and i in [15,16,17,18,19,20,21,22,23,24]:
                        qdict[n][b][p][0][i] = 0
                        flatplots[p][0][i] = 0
                    # pass
                qdict[n][b][p].ndivide(3)
    for n in vdict:
        for s in vdict[n]:
            for l in vdict[n][s]:
                for b in vdict[n][s][l]:
                    vdict[n][s][l][b].ndivide(3)
                    for i in range(26):
                        vdict[n][s][l][b][1][i] = 0.2 * i
                        if ('Data' in f) and i in [15,16,17,18,19,20,21,22,23,24]:
                            vdict[n][s][l][b][0][i] = 0

    if not 'Data' in f:
        colors = ['red','orange','gold','green','skyblue','mediumpurple','plum']
        leg = ['bin 0','bin 1','bin 2','bin 3','bin 4','standard']
        plt.clf()
        for net in ["A","B","C"]:
            th1b.append(flatplots[f"bF{net}"].toTH1(f.split('/')[-2] + "_" + flatplots[f"bF{net}"].title))
            for subn in ["bFA","bFB","bFC"]:
            # for subn in ["bAB","bAC","bBC"]:
                for b in range(5):
                    quantsys.append(qdict[net][b][subn].toTH1(f.split('/')[-2] + "_" + qdict[net][b][subn].title + f"_Qsys{net}{b}Up"))
                    if RATIO:
                        qdict[net][b][subn][0] /= flatplots[subn][0]
                    tempdown = cp.deepcopy(flatplots[subn])
                    tempdown[0] = tempdown[0] * tempdown[0] / qdict[net][b][subn][0]
                    tempdown[0][~(tempdown[0] > 0)] = 0
                    quantsys.append(tempdown.toTH1(f.split('/')[-2] + "_" + qdict[net][b][subn].title + f"_Qsys{net}{b}Down"))
                    qdict[net][b][subn].make(htype='step',color=colors[b])
                    # tempdown.make(htype='step',color=colors[b])
                tempratio = cp.deepcopy(flatplots[subn])
                tempratio.title = f"Changed {net}"
                tempratio.fname = 'ratio/' + f.split('/')[-2] + "_" + flatplots[subn].title + f"_Qsys{net}Up"
                if RATIO:
                    tempratio[0] /= tempratio[0]
                tempratio.plot(same=True,htype='step',legend=leg,color='black')
        plt.clf()
        for n in ["AB","AC","BC"]:#,"FA","FB","FC"]:
            for sub in ["AB","AC","BC"]:
                for l in sub:
                    vdict[n][sub][l]["F"].fname = f"ratio/{f.split('/')[-2]}_plot{n}_weight{sub}_q{l}FUp"
                    vdict[n][sub][l]["F"].plot(htype='step',color='black')
                    for b in range(5):
                        tempdown = cp.deepcopy(flatplots[f"b{n}"])
                        if RATIO:
                            vdict[n][sub][l][b][0] /= flatplots[f"b{n}"][0]
                            tempdown[0] /= tempdown[0]
                        quantsys.append(vdict[n][sub][l][b].toTH1(f"{f.split('/')[-2]}_{vdict[n][sub][l][b].title}_VQsys{sub}{b}{l}Up"))
                        vdict[n][sub][l][b].make(htype='step',color=colors[b])
                        tempdown[0] = tempdown[0] * tempdown [0] / vdict[n][sub][l][b][0]
                        tempdown[0][~(tempdown[0] > 0)] = 0
                        quantsys.append(tempdown.toTH1(f"{f.split('/')[-2]}_{tempdown.title}_VQsys{sub}{b}{l}Down"))
                        # tempdown.make(htype='step',color=colors[b])
                    temphist = cp.deepcopy(flatplots[f"b{n}"])
                    temphist.title = f"From {n} with {sub} weights in {l} quantiles"
                    temphist.fname = f"ratio/{f.split('/')[-2]}_plot{n}_weight{sub}_q{l}Up"
                    if RATIO:
                        temphist[0] /= temphist[0]
                        temphist[0][~(temphist[0] > 0)] = 0
                    temphist.plot(same=True,htype='step',legend=leg,color='black')
    outdict.update({namelist[nit]:flatplots})
    nit += 1

th1s = []
for net in ["A","B","C"]:
    th1s.append(flatplots[f"sF{net}"].toTH1("SignalMC_Net"+net))
    th1s.append(flatplots[f"sF{net}"].toTH1("data_obs_Net"+net))
rfile.Write()
for elem in th1s + th1b:
    elem.SetDirectory(0)
rfile.Close()
pickle.dump(outdict,open('Snetplots/flatdict.p','wb'))