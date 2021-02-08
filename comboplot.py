#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
#import numpy as np
#from analib import Hist
namelist = ['bEnriched','bGen','TTbar','WJets','ZJets']
plotnames = ['pt', 'eta', 'mass', 'CSVV2', 'DeepB', 'msoft', 'DDBvL', 
             'H4qvs', 'npvs', 'npvsG', 'mpt', 'meta', 'mip', 'n2b1', 
             'submass1', 'submass2', 'subtau1', 'subtau2', 'nsv']
colorlist = ['red','orange','yellow','green','blue','indigo','violet']
nlen = len(namelist)
ftype = 'BG'
## Load pickled dictionaries of plots
arclist = []
for name in namelist:
    arclist.append(pickle.load(open(f"Snetplots/{name}.p",'rb')))
arcdata = pickle.load(open("Dnetplots/ParkedSkim.p",'rb'))
## Generate an index list
ilist = []
for i in range(nlen):
    ilist.append(i)
## Re-arrange the index list from lowest to highest contribution
isorted=False
while(not isorted):
    isorted = True
    for i in range(1,nlen):
        if arclist[ilist[i-1]]['vplots'][f"{ftype}pt"][0].sum() > arclist[ilist[i]]['vplots'][f"{ftype}pt"][0].sum():
            ilist[i-1],ilist[i] = ilist[i],ilist[i-1]
            isorted=False
## Create a dictionary of plots, containing lists of increasingly stacked plots for each plot type
combodict = {}
for pname in plotnames:
    combodict.update({pname:[arclist[ilist[0]]['vplots'][ftype+pname]]})
    for i in range(1,nlen):
        temphist = arclist[ilist[i]]['vplots'][ftype+pname]
        temphist.add(combodict[pname][i-1])
        combodict[pname].append(temphist)
    arcdata['vplots'][f"SG{pname}"].fname = f"Comboplots/C{pname}"
    arcdata['vplots'][f"SG{pname}"].xlabel = arclist[0]['vplots'][pname].xlabel
    arcdata['vplots'][f"SG{pname}"].ylabel = 'events'
## Generate a legend for the upcoming plots
leg = []
for i in ilist[::-1]:
    leg.append(namelist[i])
leg.append('Parked data')
    
## Plot each layer of plots, from back to front
for pname in plotnames:
    plt.clf()
    for layer in range(nlen-1,-1,-1):
        combodict[pname][layer].make(color=colorlist[layer],htype='bar')
    arcdata['vplots'][f"SG{pname}"].plot(same=True,legend=leg,color='black',htype='err')
    #combodict[pname][0].plot(same=True,legend=leg,color=colorlist[0],htype='bar')