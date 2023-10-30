#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import copy as cp
import numpy as np
import sys, os
from analib import Hist as histl, Hist2d
# from analib import th1f as TH1F
# from analib import tfile as TFile
# from ROOT import TFile
class Hist(histl):
    pass

base="Base"
RATIO=True

#import numpy as np
#from analib import Hist
namelist = ['bEnriched','bGen','bInc','TTbar','WJets','ZJets']
plotnames = ['pt', 'eta', 'CSVV2', 'DeepB', 'nsv',   
             'H4qvs', 'n2b1', 'submass1', 'subtau1',  
              'mass',  'msoft', 'submass2', 'subtau2', 'DDBvL',
               'phi', 'npvs', 'npvsG', 'Dist']

for net in ["A","B","C","F"]:
    if (net=="F") or (net=="A"): cntvar = 'CSVV2'
    elif net=="B": cntvar = 'H4qvs'
    elif net=="C": cntvar = 'DDBvL'
    else: raise ValueError("Invalid Net Specified")
    
    plotnames += ['metpt','nlep','mpt','meta','mjetdr','mminiPFRelIso_all','msip3d']
    plotnames += ['ept','eeta','ejetdr','eminiPFRelIso_all','esip3d']
    colorlist = ['red','orange','yellow','green','skyblue','mediumpurple']#,'plum']
    colorlist.reverse()
    colordict = {}
    for i in range(len(namelist)):
        colordict.update({namelist[i]:colorlist[i]})
        
    nlen = len(namelist)
    ## Load pickled dictionaries of plots
    # flatdict = pickle.load(open('Snetplots/flatdict.p','rb'))
    flatnames = []#'FA','FB','FC','AB','AC','BC']
    # for name in namelist + ['Data']:
    #     for key in ['sFA','sFB','sFC','sAB','sAC','sBC','bFA','bFB','bFC','bAB','bAC','bBC']:
    #         if key[0] == 's': flatdict[name][f"SG{key[1:]}"] = flatdict[name].pop(key)
    #         else: flatdict[name][f"BG{key[1:]}"] = flatdict[name].pop(key)
    arclist = []
    for name in namelist:
        arclist.append(pickle.load(open(f"Snetplots/{base}/GGH_HPT vs {name} {net}.p",'rb')))
    arcdata = pickle.load(open(f"Dnetplots/{base}/JetHT vs Combined QCD {net}.p",'rb'))
    arcdata['vplots'].update({'SGDist':arcdata['plots']['DistSte']})
    arcdata['vplots']['SGDist'][0] *= arcdata['vplots'][f"SG{cntvar}"][0].sum()
    # arcdata['vplots'].update(flatdict['Data'])
    ## Generate an index list
    ilist = []
    for i in range(nlen):
        ilist.append(i)
        arclist[i]['vplots'].update({"BGDist":arclist[i]['plots']['DistBte']})
        arclist[i]['vplots'].update({"SGDist":arclist[i]['plots']['DistSte']})
        arclist[i]['vplots']["BGDist"][0] *= arclist[i]['vplots'][f"BG{cntvar}"][0].sum()
        arclist[i].update({'color':colorlist[i]})
        # arclist[i]['vplots'].update(flatdict[namelist[i]])
    ## Re-arrange the index list from lowest to highest contribution
    isorted=False
    while(not isorted):
        isorted = True
        for i in range(1,nlen):
            if arclist[ilist[i-1]]['vplots'][f"BG{cntvar}"][0].sum() > arclist[ilist[i]]['vplots'][f"BG{cntvar}"][0].sum():
                ilist[i-1],ilist[i] = ilist[i],ilist[i-1]
                isorted=False
    
    ## Create a dictionary of plots, containing lists of increasingly stacked plots for each plot type
    combodict = {}
    for pname in plotnames + flatnames:
        combodict.update({pname:[cp.deepcopy(arclist[ilist[0]]['vplots']["BG"+pname])]})
        for i in range(1,nlen):
            temphist = cp.deepcopy(arclist[ilist[i]]['vplots']["BG"+pname])
    
            # np.nan_to_num(temphist[0])
            # np.nan_to_num()
    
            temphist.add(combodict[pname][i-1])
            combodict[pname].append(temphist)
        arcdata['vplots'][f"SG{pname}"].fname = f"Comboplots/C{pname}"
        if pname != 'Dist' and pname not in flatnames:
            arcdata['vplots'][f"SG{pname}"].xlabel = arclist[0]['vplots'][pname].xlabel
            arcdata['vplots'][f"SG{pname}"].ylabel = 'events'
        elif pname in flatnames:
            arcdata['vplots'][f"SG{pname}"].ylabel = 'events'
        else:
            arcdata['vplots'][f"SG{pname}"].xlabel = "Confidence"
            arcdata['vplots'][f"SG{pname}"].ylabel = 'Events'
            arcdata['vplots'][f"SG{pname}"].ser *= 0
            
    comboc = [colordict[namelist[ilist[0]]]]
    for i in range(1,nlen):
        comboc.append(colordict[namelist[ilist[i]]])
        
    ## Generate a dictionary of ratio plots
    ratiodict = {}
    for pname in plotnames + flatnames:
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
    for pname in plotnames + flatnames:
        sigdict.update({pname: cp.deepcopy(arclist[0]['vplots']["SG"+pname])})
        sigdict[pname].mult = np.sum(combodict[pname][-1][0]) / np.sum(sigdict[pname][0])
        sigdict[pname].ival = sigdict[pname][0].sum()
        sigdict[pname][0] *= sigdict[pname].mult
        sigdict[pname].xlabel, sigdict[pname].ylabel, sigdict[pname].fname = '','',''
    
    bgnum, signum = 0,0
    # ## Create ROOT file for Combine
    # os.remove('Combined.root')
    # rfile = TFile('Combined.root','UPDATE')
    # th1d = arcdata['plots']['DistSte'].toTH1('data_obs', arcdata['vplots'][f"SG{cntvar}"}][0].sum())
    # th1s = arclist[0]['plots']['DistSte'].toTH1('SignalMC', arclist[0]['vplots'][f"SG{cntvar}"][0].sum())
    # signum = arclist[0]['plots']['DistSte'][0][-10:].sum()*arclist[0]['vplots'][f"SG{cntvar}"][0].sum()
    # datnum = arcdata['plots']['DistSte'][0][-10:].sum()*arcdata['vplots'][f"SG{cntvar}"][0].sum()
    # hdict = {}
    # for i in ilist:
    #     hdict[i] = arclist[i]['plots']['DistBte'].toTH1(namelist[i]+'BG', arclist[i]['vplots'][f"BG{cntvar}"][0].sum())
    #     bgnum += arclist[i]['plots']['DistBte'][0][-10:].sum() * arclist[i]['vplots'][f"BG{cntvar}"][0].sum()
    
    # rfile.Write()
    # # th1d.SetDirectory(0)
    # # th1s.SetDirectory(0)
    # # for key in hdict:
    # #     hdict[key].SetDirectory(0)
    # rfile.Close()
    
    
    ## Generate a legend for the upcoming plots
    leg = []
    leg.append(f"GGH MC ({round(sigdict[cntvar].ival)} * {round(sigdict[cntvar].mult)})")
    leg.append(f"JetHT data ({round(arcdata['vplots']['SG'+cntvar][0].sum())})")
    leglist = ['QCDbEnriched','QCDbGen','QCDInc','TTbar','WJets','ZJets']
    for i in ilist[::-1]:
        leg.append(f"{leglist[i]} ({round(arclist[i]['vplots']['BG'+cntvar][0].sum())})")
    
    
    
    ## Plot each layer of plots, from back to front
    lv = False
    if RATIO:
        for pname in plotnames + ["DistL"] + flatnames:
            if pname == "DistL":
                pname = "Dist"
                arcdata['vplots']["SGDist"].fname += "L"
                # ratiodict['Dist']["Dist"].fname += "L"
                lv = 'set'
            arcdata['vplots'][f"SG{pname}"].fname += f"_{net}"
            plt.clf()
            fig, axis = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
            if np.nan_to_num(sigdict[pname][0]).sum() == 0: continue
            for layer in range(nlen-1,-1,-1):
                combodict[pname][layer].make(color=comboc[layer],htype='bar',logv=lv,parent=axis[0])
            sigdict[pname].make(color='r',htype='step',logv=lv,parent=axis[0])
            ratiodict[pname].xlabel = arcdata['vplots'][f"SG{pname}"].xlabel
            ratiodict[pname].plot(same=True,color='k',htype='err',parent=axis[1],clean=True,ylim=[-1,1])
            arcdata['vplots'][f"SG{pname}"].plot(same=True,legend=leg,lloc=1,color='k',htype='err',logv=lv,parent=axis[0], lsize=12)
    else:
        for pname in plotnames + ["DistL"] + flatnames:
            if pname == "DistL":
                pname = "Dist"
                arcdata['vplots']["SGDist"].fname += "L"
                # ratiodict['Dist']["Dist"].fname += "L"
                lv = 'set'
            arcdata['vplots'][f"SG{pname}"].fname += f"_{net}"
            plt.clf()
            # fig, axis = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
            if np.nan_to_num(sigdict[pname][0]).sum() == 0: continue
            for layer in range(nlen-1,-1,-1):
                combodict[pname][layer].make(color=comboc[layer],htype='bar',logv=lv)
            sigdict[pname].make(color='r',htype='step',logv=lv)
            # ratiodict[pname].plot(same=True,color='k',htype='err',parent=axis[1],clean=True,ylim=[-1,1])
            arcdata['vplots'][f"SG{pname}"].plot(same=True,legend=leg,lloc=1,color='k',htype='err',\
                                                 lsize=14,xsize=30,ysize=30)




    #combodict[pname][0].plot(same=True,legend=leg,color=colorlist[0],htype='bar')