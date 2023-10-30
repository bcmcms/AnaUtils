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
import libcppyy3_6 as libcppyy

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


# useF = True
nbin = 17
RATIO = False
path="Snetplots"
bgslice = ['Combined QCD','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
namelist = ['Full','bEnriched','bGen','bInc','TTbar','WJets','ZJets','Data']
sysvars = ['pt','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs','submass1','submass2','nsv','subtau1','subtau2','n2b1','Trig']
nit = 0

# folders = ["Diststore/QCDGen/"]
## Create ROOT file for Combine
# os.remove(f"SubNet_CombinedX{nbin}.root")
rfile = TFile(f"SubNet_CombinedY{nbin}.root",'RECREATE')
th1b, quantsys, th1s = [], [], []
## controls weighting amount for shape uncertainties





isospearmandict = pickle.load(open(f"{path}/isospearmandictY.p",'rb'))
BLIND = isospearmandict['BLIND']
REGION = isospearmandict['REGION']
#%%
for bg in bgslice:
    if   bg == 'Combined QCD': fn = 'Full'
    elif bg == 'bEnriched': fn = 'QCDEnr'
    elif bg == 'bGen': fn = 'QCDGen'
    elif bg == 'bInc': fn = 'QCDInc'
    else: fn = bg
    
    qdict = isospearmandict[fn]['qdict']
    vdict = isospearmandict[fn]['vdict']
    flatplots = isospearmandict[fn]['flatplots']
    distplots = isospearmandict[fn]['distplots']
    for a in qdict:
        for b in qdict[a]:
            for c in qdict[a][b]:
                for d in qdict[a][b][c]:
                    if BLIND and "F" in c: 
                        qdict[a][b][c][d][0] = np.delete(qdict[a][b][c][d][0],[-1,-2])
                        qdict[a][b][c][d][1] = np.delete(qdict[a][b][c][d][1],[-1,-2])
                        qdict[a][b][c][d].ser = np.delete(qdict[a][b][c][d].ser,[-1,-2])
                    qdict[a][b][c][d].toTH1 = toTH1.__get__(qdict[a][b][c][d])
    for a in vdict:
        for b in vdict[a]:
            for c in vdict[a][b]:
                for d in vdict[a][b][c]:
                    for e in vdict[a][b][c][d]:
                        if BLIND and "F" in a:
                            vdict[a][b][c][d][e][0] = np.delete(vdict[a][b][c][d][e][0],[-1,-2])
                            vdict[a][b][c][d][e][1] = np.delete(vdict[a][b][c][d][e][1],[-1,-2])
                            vdict[a][b][c][d][e].ser = np.delete(vdict[a][b][c][d][e].ser,[-1,-2])
                            
                        vdict[a][b][c][d][e].toTH1 = toTH1.__get__(vdict[a][b][c][d][e])
    for a in flatplots:
        if BLIND and "F" in a:
            flatplots[a][0] = np.delete(flatplots[a][0],[-1,-2])
            flatplots[a][1] = np.delete(flatplots[a][1],[-1,-2])
            flatplots[a].ser = np.delete(flatplots[a].ser,[-1,-2])
        flatplots[a].toTH1 = toTH1.__get__(flatplots[a])
    for a in distplots:
        # if BLIND:
        #     distplots[a][0][15]=0
        #     distplots[a][0][16]=0
        #     distplots[a].ser[15]=0
        #     distplots[a].ser[16]=0
        distplots[a].toTH1 = toTH1.__get__(distplots[a])
    

    if not 'Data' in fn:
        bslice = fn
        colors = ['red','orange','gold','green','skyblue','mediumpurple','plum']
        leg = ['bin 0','bin 1','bin 2','bin 3','bin 4','standard']
        plt.clf()
        for net in ["A","B","C"]:
            for subn in ["bFA","bFB","bFC"] + ["bAB","bAC","bBC"]:
                for b in range(5):
                    # quantsys.append(qdict[net][b][subn]['U'].toTH1(bslice + "_" + qdict[net][b][subn]['U'].title + f"_Qsys{net}{b}_{bslice}Up"))
                    # quantsys.append(qdict[net][b][subn]['D'].toTH1(bslice + "_" + qdict[net][b][subn]['D'].title + f"_Qsys{net}{b}_{bslice}Down"))
                    quantsys.append(qdict[net][b][subn]['U'].toTH1(bslice + "_" + qdict[net][b][subn]['U'].title + f"_Qsys{net}{b}Up"))
                    qdict[net][b][subn]['U'].make(color=colors[b],htype='step')
                    quantsys.append(qdict[net][b][subn]['D'].toTH1(bslice + "_" + qdict[net][b][subn]['D'].title + f"_Qsys{net}{b}Down"))
                thist = cp.deepcopy(flatplots[subn])
                thist.title = f"{bg} {subn[1]} vs {subn[2]} with raised {net}"
                thist.fname = f"ratio/{bslice}_{qdict[net][b][subn]['D'].title}_Qsys{net}Up"
                thist.plot(same=True,htype='step',color='black',legend=['Q1 Up','Q2 Uo','Q3 Up','Q4 Up','Q5 Up',],ylim=[0,flatplots[subn][0].max()*1.2],xsize=20,ysize=20,tsize=20,lsize=16)
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
                vdict[n][sub][sub[0]]["F"]['U'].make(color='red',htype='step')
                vdict[n][sub][sub[0]]["F"]['D'].make(color='blue',htype='step')
                thist = cp.deepcopy( flatplots[f"b{n}"])
                thist.title = f"{bg} {tstr} with 2D systematic"
                thist.fname = f"ratio/{bslice}_{tstr}_{sub}2D"
                thist.plot(same=True,htype='step',color='black',legend=[f"1σ up {sub}2D",f"1σ down {sub}2D"],ylim=[0,flatplots[f"b{n}"][0].max()*1.4],xsize=20,ysize=20,tsize=20,lsize=16,lloc=4)
                
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
    nit += 1
    if 'Data' in fn:
        for net in ["A","B","C"]:
            th1s.append(flatplots[f"sF{net}"].toTH1("data_obs_Net"+net,zerofill=False))
        for n in ["AB","AC","BC"]:
            th1s.append(flatplots[f"s{n}"].toTH1(f"data_obs_{n[-1]}in{n[0]}",zerofill=False))
    elif 'Full' in fn:
        for net in ["A","B","C"]:
            th1s.append(flatplots[f"sF{net}"].toTH1("SignalMC_Net"+net,zerofill=.001))
        for nets in ["AB","AC","BC"]:
            th1s.append(flatplots[f"s{nets}"].toTH1(f"SignalMC_{nets[1]}in{nets[0]}",zerofill=.001))
        if not REGION:
            varsysdict = isospearmandict['Full']['varsysdict']
            for v in sysvars:
                for n in ["A","B","C"]:
                    for d in ["Up","Down"]:
                        if BLIND: 
                            varsysdict[v][n][d][0][15] = 0
                            varsysdict[v][n][d][0][16] = 0
                        varsysdict[v][n][d].toTH1 = toTH1.__get__(varsysdict[v][n][d])
                        th1s.append(varsysdict[v][n][d].toTH1(f"SignalMC_Net{n}_{v}{d}",zerofill=.001))
                    varsysdict[v][n]['Up'].make(color='red',htype='step')
                    varsysdict[v][n]['Down'].make(color='blue',htype='step')
                    thist = cp.deepcopy( flatplots[f"sF{n}"])
                    thist.title = f"Combined MC Net {n} signal {v} systematic"
                    thist.fname = f"ratio/{bslice}_Net{n}_{v}"
                    thist.plot(same=True,htype='step',color='black',legend=[f"1σ up {v}",f"1σ down {v}"],ylim=[0,flatplots[f"sF{n}"][0].max()*1.4],xsize=20,ysize=20,tsize=20,lsize=16,lloc=4)
        

    
rfile.Write()
for elem in th1s + th1b:
    elem.SetDirectory(0)
# rfile.Close()