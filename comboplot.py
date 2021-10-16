#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import copy as cp
import numpy as np
import sys, os

# from analib import th1f as TH1F
# from analib import tfile as TFile

from ROOT import TFile

#import numpy as np
#from analib import Hist
namelist = ['bEnriched','bGen','TTbar','WJets','ZJets','QCDinclusive']
plotnames = ['pt', 'eta', 'phi', 'mass', 'CSVV2', 'DeepB', 'msoft', 'DDBvL',
             'H4qvs', 'npvs', 'npvsG', 'mpt', 'meta', 'mip', 'n2b1',
             'submass1', 'submass2', 'subtau1', 'subtau2', 'nsv','Dist']
if True:
    plotnames += ['metpt','nlep','mpt','meta','mjetdr','mminiPFRelIso_all','msip3d',
                  'ept','eeta','ejetdr','eminiPFRelIso_all','esip3d']
colorlist = ['red','orange','yellow','green','skyblue','mediumpurple','plum']
nlen = len(namelist)
## Load pickled dictionaries of plots
arclist = []
for name in namelist:
    arclist.append(pickle.load(open(f"Snetplots/GGH_HPT vs {name}.p",'rb')))
arcdata = pickle.load(open("Dnetplots/JetHT vs Combined QCD.p",'rb'))
arcdata['vplots'].update({'SGDist':arcdata['plots']['DistSte']})
arcdata['vplots']['SGDist'][0] *= arcdata['vplots']["SGCSVV2"][0].sum()
## Generate an index list
ilist = []
for i in range(nlen):
    ilist.append(i)
    arclist[i]['vplots'].update({"BGDist":arclist[i]['plots']['DistBte']})
    arclist[i]['vplots'].update({"SGDist":arclist[i]['plots']['DistSte']})
    arclist[i]['vplots']["BGDist"][0] *= arclist[i]['vplots']['BGCSVV2'][0].sum()
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

        # np.nan_to_num(temphist[0])
        # np.nan_to_num()

        temphist.add(combodict[pname][i-1])
        combodict[pname].append(temphist)
    arcdata['vplots'][f"SG{pname}"].fname = f"Comboplots/C{pname}"
    if pname != 'Dist':
        arcdata['vplots'][f"SG{pname}"].xlabel = arclist[0]['vplots'][pname].xlabel
        arcdata['vplots'][f"SG{pname}"].ylabel = 'events'
    else:
        arcdata['vplots'][f"SG{pname}"].xlabel = "Confidence"
        arcdata['vplots'][f"SG{pname}"].ylabel = 'Fractional Distribution'
        arcdata['vplots'][f"SG{pname}"].ser *= 0


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
ratiodict["Dist"].ser *= 0

## Prepare the Signal MC
sigdict = {}
for pname in plotnames:
    sigdict.update({pname: cp.deepcopy(arclist[0]['vplots']["SG"+pname])})
    sigdict[pname].mult = np.sum(combodict[pname][-1][0]) / np.sum(sigdict[pname][0])
    sigdict[pname].ival = sigdict[pname][0].sum()
    sigdict[pname][0] *= sigdict[pname].mult
    sigdict[pname].xlabel, sigdict[pname].ylabel, sigdict[pname].fname = '','',''

bgnum, signum = 0,0
# ## Create ROOT file for Combine
# os.remove('Combined.root')
# rfile = TFile('Combined.root','UPDATE')
# th1d = arcdata['plots']['DistSte'].toTH1('data_obs', arcdata['vplots']['SGCSVV2'][0].sum())
# th1s = arclist[0]['plots']['DistSte'].toTH1('SignalMC', arclist[0]['vplots']['SGCSVV2'][0].sum())
# signum = arclist[0]['plots']['DistSte'][0][-10:].sum()*arclist[0]['vplots']['SGCSVV2'][0].sum()
# datnum = arcdata['plots']['DistSte'][0][-10:].sum()*arcdata['vplots']['SGCSVV2'][0].sum()
# hdict = {}
# for i in ilist:
#     hdict[i] = arclist[i]['plots']['DistBte'].toTH1(namelist[i]+'BG', arclist[i]['vplots']['BGCSVV2'][0].sum())
#     bgnum += arclist[i]['plots']['DistBte'][0][-10:].sum() * arclist[i]['vplots']['BGCSVV2'][0].sum()

# rfile.Write()
# # th1d.SetDirectory(0)
# # th1s.SetDirectory(0)
# # for key in hdict:
# #     hdict[key].SetDirectory(0)
# rfile.Close()


## Generate a legend for the upcoming plots
leg = []
leg.append(f"GGH MC ({round(sigdict['CSVV2'].ival)} * {round(sigdict['CSVV2'].mult)})")
leg.append(f"JetHT data ({round(arcdata['vplots']['SGCSVV2'][0].sum())})")
for i in ilist[::-1]:
    leg.append(f"{namelist[i]} ({round(arclist[i]['vplots']['BGCSVV2'][0].sum())})")



## Plot each layer of plots, from back to front
lv = False
for pname in plotnames + ["DistL"]:
    if pname == "DistL":
        pname = "Dist"
        arcdata['vplots']["SGDist"].fname += "L"
        # ratiodict['Dist']["Dist"].fname += "L"
        lv = 'set'
    plt.clf()
    fig, axis = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
    for layer in range(nlen-1,-1,-1):
        combodict[pname][layer].make(color=colorlist[layer],htype='bar',parent=axis[0],logv=lv)
    sigdict[pname].make(color='r',htype='step',parent=axis[0],logv=lv)
    ratiodict[pname].plot(same=True,color='k',htype='err',parent=axis[1],clean=True,ylim=[-0.5,0.5])
    arcdata['vplots'][f"SG{pname}"].plot(same=True,legend=leg,color='k',htype='err',parent=axis[0],logv=lv)




    #combodict[pname][0].plot(same=True,legend=leg,color=colorlist[0],htype='bar')