#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import copy as cp
import numpy as np
#import numpy as np
#from analib import Hist
namelist = ['bEnriched','bGen','TTbar','WJets','ZJets','QCDinclusive']
plotnames = ['pt', 'eta', 'phi', 'mass', 'CSVV2', 'DeepB', 'msoft', 'DDBvL', 
             'H4qvs', 'npvs', 'npvsG', 'mpt', 'meta', 'mip', 'n2b1', 
             'submass1', 'submass2', 'subtau1', 'subtau2', 'nsv']#,
             #'mmsum','mptsum','metasum','mqmpt','mqppt','mqmeta','mqpeta','mqmip3d','mqpip3d','mqmsip3d','mqpsip3d','MuSumvJetpT','MuQmvJetpT','MuQpvJetpT']
colorlist = ['red','orange','yellow','green','skyblue','mediumpurple','plum']
nlen = len(namelist)
## Load pickled dictionaries of plots
arclist = []
for name in namelist:
    arclist.append(pickle.load(open(f"Snetplots/GGH_HPT vs {name}.p",'rb')))
arcdata = pickle.load(open("Dnetplots/JetHT vs Combined QCD.p",'rb'))
## Generate an index list
ilist = []
for i in range(nlen):
    ilist.append(i)
## Re-arrange the index list from lowest to highest contribution
isorted=False
while(not isorted):
    isorted = True
    for i in range(1,nlen):
        if arclist[ilist[i-1]]['vplots']["BGpt"][0].sum() > arclist[ilist[i]]['vplots']["BGpt"][0].sum():
            ilist[i-1],ilist[i] = ilist[i],ilist[i-1]
            isorted=False

## Create a dictionary of plots, containing lists of increasingly stacked plots for each plot type
combodict = {}
for pname in plotnames:
    combodict.update({pname:[cp.deepcopy(arclist[ilist[0]]['vplots']["BG"+pname])]})
    for i in range(1,nlen):
        temphist = cp.deepcopy(arclist[ilist[i]]['vplots']["BG"+pname])
        temphist.add(combodict[pname][i-1])
        combodict[pname].append(temphist)
    arcdata['vplots'][f"SG{pname}"].fname = f"Comboplots/C{pname}"
    arcdata['vplots'][f"SG{pname}"].xlabel = arclist[0]['vplots'][pname].xlabel
    arcdata['vplots'][f"SG{pname}"].ylabel = 'events'

## Generate a dictionary of ratio plots
ratiodict = {}
for pname in plotnames:
    temphist = cp.deepcopy(arcdata['vplots'][f"SG{pname}"])#combodict[pname][-1])
    temphist = temphist.sub(combodict[pname][-1],split=True).divideby(combodict[pname][-1])
    temphist.fname = ''
    temphist.xlabel=''
    temphist.ylabel=''
    temphist.ser = 1/arcdata['vplots'][f"SG{pname}"][0]
    temphist.ser[np.isinf(temphist.ser)] = np.nan
    ratiodict.update({pname:temphist})
    
## Prepare the Signal MC
sigdict = {}
for pname in plotnames:
    sigdict.update({pname: cp.deepcopy(arclist[0]['vplots']["SG"+pname])})
    sigdict[pname].mult = np.sum(combodict[pname][-1][0]) / np.sum(sigdict[pname][0])
    sigdict[pname].ival = sigdict[pname][0].sum()
    sigdict[pname][0] *= sigdict[pname].mult
    sigdict[pname].xlabel, sigdict[pname].ylabel, sigdict[pname].fname = '','',''

## Generate a legend for the upcoming plots
leg = []
leg.append(f"GGH MC ({round(sigdict['CSVV2'].ival)} * {round(sigdict['CSVV2'].mult)})")
leg.append(f"JetHT data ({round(arcdata['vplots']['SGCSVV2'][0].sum())})")
for i in ilist[::-1]:
    leg.append(f"{namelist[i]} ({round(arclist[i]['vplots']['BGCSVV2'][0].sum())})")

    
## Plot each layer of plots, from back to front
for pname in plotnames:
    plt.clf()
    fig, axis = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
    for layer in range(nlen-1,-1,-1):
        combodict[pname][layer].make(color=colorlist[layer],htype='bar',parent=axis[0])
    sigdict[pname].make(color='r',htype='step',parent=axis[0])
    ratiodict[pname].plot(same=True,color='k',htype='err',parent=axis[1],clean=True,ylim=[-0.5,0.5])
    arcdata['vplots'][f"SG{pname}"].plot(same=True,legend=leg,color='k',htype='err',parent=axis[0])
    
    #combodict[pname][0].plot(same=True,legend=leg,color=colorlist[0],htype='bar')