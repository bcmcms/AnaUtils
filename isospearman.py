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

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)

#evtlist = [35899001,24910172,106249475,126514437,43203653,27186346,17599588,64962950,61283040,54831588]

class Hist(HistL):
    def toTH1(s,title,scale=1):
        th1 = TH1F(title,title,len(s.hs[0]),s.hs[1][0],s.hs[1][-1])
        for i in range(len(s.hs[0])):
            th1.SetBinContent(i,s.hs[0][i]*scale)
            th1.SetBinError(i,np.sqrt(s.ser[i])*scale)
        return th1

    def errtoTH1(s,scale=1):
        return np.histogram(s.hs[1][12:-5],s.hs[1][12:-4],weights=np.sqrt(s.ser[12:-4])*scale)


#%%

useF = True

if useF:
    folders = ["Diststore/Full/","Diststore/QCDEnr/","Diststore/QCDGen/","Diststore/QCDInc/",
               "Diststore/TTbar/","Diststore/ZJets/","Diststore/WJets/"]
    # folders = ["Diststore/QCDGen/"]
    ## Create ROOT file for Combine
    os.remove('SubNet_Combined.root')
    rfile = TFile('SubNet_Combined.root','UPDATE')
    th1b = []
else: folders = [""]

for f in folders:
    sigframe, bgframe = DataFrame(), DataFrame()
    sigframe['A'] = pickle.load(open(f"{f}Astt.p",'rb'))[:,0]
    sigframe['B'] = pickle.load(open(f"{f}Bstt.p",'rb'))
    sigframe['C'] = pickle.load(open(f"{f}Cstt.p",'rb'))[:,0]
    bgframe['A'] = pickle.load(open(f"{f}Abtt.p",'rb'))[:,0]
    bgframe['B'] = pickle.load(open(f"{f}Bbtt.p",'rb'))
    bgframe['C'] = pickle.load(open(f"{f}Cbtt.p",'rb'))[:,0]
    if useF:
        sigframe['F'] = pickle.load(open(f"{f}Fstt.p",'rb'))[:,0]
        bgframe['F'] = pickle.load(open(f"{f}Fbtt.p",'rb'))[:,0]


    if True:
        Adat = pickle.load(open(f"{f}A.p",'rb'))
        Bdat = pickle.load(open(f"{f}B.p",'rb'))
        Cdat = pickle.load(open(f"{f}C.p",'rb'))
        Fdat = pickle.load(open(f"{f}F.p",'rb'))

        for elem in [Adat, Bdat, Cdat, Fdat]:
            elem["SensB"] = [0] + list(elem["SensB"]) + [1]
            elem["SensS"] = list(elem["SensS"])
            if useF:
                for i in range(1,6):
                    elem["SensB"].pop(i)
                    elem["SensS"].pop(i)
        if useF:
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
                }
            flatplots = {
                "sFA": Hist(abbounds,None,"20%-quantile A occupancy","Fractional Distribution","FAflatS","FA_signal"),
                "sFB": Hist(bbbounds,None,"20%-quantile B occupancy","Fractional Distribution","FBflatS","FB_signal"),
                "sFC": Hist(cbbounds,None,"20%-quantile C occupancy","Fractional Distribution","FCflatS","FC_signal"),
                "bFA": Hist(abbounds,None,"20%-quantile A occupancy","Fractional Distribution","FAflatB","NetA"),
                "bFB": Hist(bbbounds,None,"20%-quantile B occupancy","Fractional Distribution","FBflatB","NetB"),
                "bFC": Hist(cbbounds,None,"20%-quantile C occupancy","Fractional Distribution","FCflatB","NetC"),
                }

            sigframe["W"] = Fdat["WS"].reset_index(drop=True)
            bgframe["W"] = Fdat["WB"].reset_index(drop=True)
            # sigframe["Abin"], sigframe["Bbin"], sigframe["Cbin"] = 5, 5, 5
            # bgframe["Abin"], bgframe["Bbin"], bgframe["Cbin"] = 5, 5, 5
            sigframe ["Fbin"], bgframe["Fbin"] = 5, 5

            for i in range(5,0,-1):
                # sigframe["Abin"][sigframe["A"] <= Adat["SensB"][i]] = i-1
                # sigframe["Bbin"][sigframe["B"] <= Bdat["SensB"][i]] = i-1
                # sigframe["Cbin"][sigframe["C"] <= Cdat["SensB"][i]] = i-1
                sigframe["Fbin"][sigframe["F"] <= Fdat["SensS"][i]] = i-1
                # bgframe["Abin"][bgframe["A"] <= Adat["SensB"][i]] = i-1
                # bgframe["Bbin"][bgframe["B"] <= Bdat["SensB"][i]] = i-1
                # bgframe["Cbin"][bgframe["C"] <= Cdat["SensB"][i]] = i-1
                bgframe["Fbin"][bgframe["F"] <= Fdat["SensS"][i]] = i-1
            for i in range(5):
                flatplots["sFA"].fill(sigframe["A"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
                flatplots["sFB"].fill(sigframe["B"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
                flatplots["sFC"].fill(sigframe["C"][sigframe["Fbin"] == i] + i, sigframe["W"][sigframe["Fbin"] == i])
                flatplots["bFA"].fill(bgframe["A"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
                flatplots["bFB"].fill(bgframe["B"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])
                flatplots["bFC"].fill(bgframe["C"][bgframe["Fbin"] == i] + i, bgframe["W"][bgframe["Fbin"] == i])

            bounds = [abbounds, bbbounds, cbbounds]
            qdict = {"A":{},"B":{},"C":{}}
            net = ["A","B","C"]
            for n in range(3):
                for b in range(5):
                    qdict.update({b:{}})
                    tempweights = Adat['WS']
                    tempweights[np.logical_and(tempweights < bounds[n][b], tempweights >= bounds[n][b+1])] *= 1.5
                    for key in distplots:
                        qdict[net[n]].update({key:distplots[key]})
                    qdict[net[n]][b]["sFA"].fill(sigframe["F"],sigframe["A"],tempweights)
                    qdict[net[n]][b]["sFB"].fill(sigframe["F"],sigframe["B"],tempweights)
                    qdict[net[n]][b]["sFC"].fill(sigframe["F"],sigframe["C"],tempweights)
                    qdict[net[n]][b]["bFA"].fill(bgframe["F"],bgframe["A"],tempweights)
                    qdict[net[n]][b]["bFB"].fill(bgframe["F"],bgframe["B"],tempweights)
                    qdict[net[n]][b]["bFC"].fill(bgframe["F"],bgframe["C"],tempweights)


            distplots["sFA"].fill(sigframe["F"],sigframe["A"],Adat['WS'])
            distplots["sFB"].fill(sigframe["F"],sigframe["B"],Adat['WS'])
            distplots["sFC"].fill(sigframe["F"],sigframe["C"],Adat['WS'])
            distplots["bFA"].fill(bgframe["F"],bgframe["A"],Adat['WB'])
            distplots["bFB"].fill(bgframe["F"],bgframe["B"],Adat['WB'])
            distplots["bFC"].fill(bgframe["F"],bgframe["C"],Adat['WB'])



        else:
            distplots = {
                "sAB": Hist2D([Adat['SensS'],Bdat['SensS']],None,"Subnet A signal","Subnet B signal","ABsubdistS"),
                "sAC": Hist2D([Adat['SensS'],Cdat['SensS']],None,"Subnet A signal","Subnet C signal","ACsubdistS"),
                "sBC": Hist2D([Bdat['SensS'],Cdat['SensS']],None,"Subnet B signal","Subnet C signal","BCsubdistS"),
                "bAB": Hist2D([Adat['SensB'],Bdat['SensB']],None,"Subnet A BG","Subnet B BG","ABsubdistB"),
                "bAC": Hist2D([Adat['SensB'],Cdat['SensB']],None,"Subnet A BG","Subnet C BG","ACsubdistB"),
                "bBC": Hist2D([Bdat['SensB'],Cdat['SensB']],None,"Subnet B BG","Subnet C BG","BCsubdistB"),
                }

            distplots["sAB"].fill(sigframe["A"],sigframe["B"],Adat['WS'])
            distplots["sAC"].fill(sigframe["A"],sigframe["C"],Adat['WS'])
            distplots["sBC"].fill(sigframe["B"],sigframe["C"],Adat['WS'])
            distplots["bAB"].fill(bgframe["A"],bgframe["B"],Adat['WB'])
            distplots["bAC"].fill(bgframe["A"],bgframe["C"],Adat['WB'])
            distplots["bBC"].fill(bgframe["B"],bgframe["C"],Adat['WB'])

    else:
        bounds = (0,2)#(0.2,0.8)
        bins = 10
        distplots = {
            "sAB": Hist2D(bins,[bounds,bounds],"Subnet A signal","Subnet B signal","ABsubdistS"),
            "sAC": Hist2D(bins,[bounds,bounds],"Subnet A signal","Subnet C signal","ACsubdistS"),
            "sBC": Hist2D(bins,[bounds,bounds],"Subnet B signal","Subnet C signal","BCsubdistS"),
            "bAB": Hist2D(bins,[bounds,bounds],"Subnet A BG","Subnet B BG","ABsubdistB"),
            "bAC": Hist2D(bins,[bounds,bounds],"Subnet A BG","Subnet C BG","ACsubdistB"),
            "bBC": Hist2D(bins,[bounds,bounds],"Subnet B BG","Subnet C BG","BCsubdistB"),
            }

        distplots["sAB"].fill(sigframe["A"],sigframe["B"])
        distplots["sAC"].fill(sigframe["A"],sigframe["C"])
        distplots["sBC"].fill(sigframe["B"],sigframe["C"])
        distplots["bAB"].fill(bgframe["A"],bgframe["B"])
        distplots["bAC"].fill(bgframe["A"],bgframe["C"])
        distplots["bBC"].fill(bgframe["B"],bgframe["C"])

    for p in distplots:
        if False:
            distplots[p][1] = [0,1,2,3,4,5,6,7,8,9,10]
            distplots[p][2] = [0,1,2,3,4,5,6,7,8,9,10]
            distplots[p][0] /= (distplots[p][0].sum()/100)
        elif useF:
            distplots[p][1] = [0,1,2,3,4,5]
            distplots[p][2] = [0,1,2,3,4,5]
            for i in range(26):
                flatplots[p][1][i] = 0.2 * i
            # for i in range(5):
            #     for j in range(5):
            #         flatplots[p][0][j+(5*i)] = distplots[p][0][i][j]
            distplots[p][0] /= (distplots[p][0].sum()/25)
            flatplots[p].ndivide(3)
            flatplots[p].plot(htype='bar',logv=False,error=True)
        distplots[p].plot(text=True,edgecolor='k',tlen=5)
    if useF:
        for net in ["A","B","C"]:
            th1b.append(flatplots[f"bF{net}"].toTH1(f.split('/')[-2] + "_" + flatplots[f"bF{net}"].title))
if useF:
    th1s = []
    for net in ["A","B","C"]:
        th1s.append(flatplots[f"sF{net}"].toTH1("SignalMC_Net"+net))
        th1s.append(flatplots[f"sF{net}"].toTH1("data_obs_Net"+net))
    rfile.Write()
    for elem in th1s + th1b:
        elem.SetDirectory(0)
    rfile.Close()

if not useF:
    datalist = []
    for frame in [sigframe, bgframe]:
        datalist.append(spearmanr(frame).correlation)
    datalist.append(datalist[0] - datalist[1])

    mindata = []
    fnames = ['IsoSpearmanSGD','IsoSpearmanBGD']
    for k in range(2):
        distdata = datalist[k] * 0
        for x in range(distdata.shape[0]):
            for y in range(distdata.shape[1]):
                distdata[y,x] = min((np.sum((datalist[k][y] - datalist[k][x])**2))**.5,
                                    (np.sum((datalist[k][y] + datalist[k][x])**2))**.5)
        mindata.append(distdata)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        linkage = hierarchy.ward(squareform(mindata[k]))
        dendro = hierarchy.dendrogram(
            linkage, labels=['A','B','C'], ax=ax1, leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro['ivl']))
        imdata = mindata[k][dendro['leaves'], :][:, dendro['leaves']]
        ax2.imshow(imdata)
        if True:
            strarray = imdata.round(3).astype(str)
            for i in range(len(imdata[0])):
                for j in range(len(imdata[1])):
                    plt.text(i,j, strarray[i,j],color="w", ha="center", va="center", fontweight='normal',fontsize=11).set_path_effects([PathEffects.withStroke(linewidth=2,foreground='k')])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'])#, rotation='vertical')
        ax2.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        #plt.show()
        plt.gcf()
        plt.savefig(fnames[k])

    fnames = ['IsoSpearmanSG','IsoSpearmanBG','IsoDiff2D']
    for k in range(3):
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 16))
        imdata = datalist[k][dendro['leaves'], :][:, dendro['leaves']]
        ax1.imshow(imdata)
        if True:
            strarray = imdata.round(3).astype(str)
            for i in range(len(imdata[0])):
                for j in range(len(imdata[1])):
                    plt.text(i,j, strarray[i,j],color="w", ha="center", va="center", fontweight='normal',fontsize=11).set_path_effects([PathEffects.withStroke(linewidth=2,foreground='k')])
        ax1.set_xticks(dendro_idx)
        ax1.set_yticks(dendro_idx)
        ax1.set_xticklabels(dendro['ivl'])#, rotation='vertical')
        ax1.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        #plt.show()
        plt.gcf()
        plt.savefig(fnames[k])