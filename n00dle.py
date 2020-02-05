#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility n00dle.py                               ###
###                                                                  ###
### Currently doesn't support options... but we're improving!        ###
########################################################################

#import ROOT as R
#R.gROOT.SetBatch(True)  ## Don't display histograms or canvases when drawn

import os
import subprocess
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import copy as cp
from analib import * 

def mc(files):
    ## This histogram object is used to accumulate and render our data, defined above
    pdgplt = Hist(40,(-0.5,39.5))
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open our file for processing
        f = uproot.open(files[fnum])
        events = f.get('Events')
        pdgid = events.array('GenPart_pdgId')
        parid = events.array('GenPart_genPartIdxMother')
        pt = events.array('GenPart_pt')
        print('Processing ' + str(len(pdgid)) + ' events')
        outlist = []
        ## Loop over pdgid, using extremely slow ROOT-like logic instead of uproot logic.
        for event in range(pdgid.size):
            for iGen in range(pdgid[event].size):
                if abs(pdgid[event][iGen]) == 5:
                    parentIdx = parid[event][iGen]
                    if parentIdx == -1: continue
                    parentId = pdgid[event][parentIdx]
                    if abs(parentId) == 9000006:
                        outlist.append(36)
                        print(str(event) + " - " + str(iGen) + " = " + str(pt[event][iGen]))

                    else:
                        outlist.append(parentId)
        ## Fill out histogram with the list of values we obtained
        pdgplt.fill(outlist)
        
    
    plt.clf()  
    plot = pdgplt.make(logv=True) 
    plt.xlabel('Parent PdgId')
    plt.ylabel('Number of b Children')
    plt.savefig('upplots/parents.png')
    plt.show()
    return plot

def ana(files):
    ## Define what pdgId we expect the A to have
    #Aid = 9000006
    Aid = 36
    ## Make a dictionary of histogram objects
    bjplots = {}
    for i in range(1,5):
        bjplots.update({
        "beta"+str(i):      Hist(66,(-3.3,3.3),'GEN b '+str(i)+' Eta','Events','upplots/beta'+str(i)),
        "bpT"+str(i):       Hist(100,(-0.5,99.5),'GEN pT of b '+str(i),'Events','upplots/bdRpT'+str(i)),
        "bjetpT"+str(i):    Hist(100,(-0.5,99.5),'Matched RECO jet '+str(i)+' pT','Events','upplots/RjetpT'+str(i)),
        "bjeteta"+str(i):   Hist(66,(-3.3,3.3),'Matched RECO jet '+str(i)+' Eta','Events','upplots/Rjeteta'+str(i))
        })
    plots = {
        "bphi":     Hist(66,(-3.3,3.3),'GEN b Phi','Events','upplots/bphi'),
        "bdR":      Hist(100,(-0.005,1.995),'GEN b to matched jet dR','Events','upplots/bdR'),
        "RjetpT":   Hist(100,(-0.5,99.5),'All RECO jet pT','Events','upplots/RjetpT'),
        "GoverRjetpT":  Hist(100,(-0.005,.995),'jet pT','Ratio of GEN b pT / RECO jet pT for matched jets','upplots/GRjetpT'),
        "bdRvlogbpT1":   Hist2d([80,100],[[-0.05,7.95],[-0.00005,.00995]],'log2(GEN b pT)','dR from 1st pT GEN b to matched RECO jet','upplots/bdRvlogbpT1'),
        "bdRvlogbpT2":   Hist2d([80,100],[[-0.05,7.95],[-0.00005,.00995]],'log2(GEN b pT)','dR from 2nd pT GEN b to matched RECO jet','upplots/bdRvlogbpT2'),
        "bdRvlogbpT3":   Hist2d([80,100],[[-0.05,7.95],[-0.00005,.00995]],'log2(GEN b pT)','dR from 3rd pT GEN b to matched RECO jet','upplots/bdRvlogbpT3'),
        "bdRvlogbpT4":   Hist2d([80,100],[[-0.05,7.95],[-0.00005,.00995]],'log2(GEN b pT)','dR from 4th pT GEN b to matched RECO jet','upplots/bdRvlogbpT4'),
        "jetoverbpTvlogbpT1":    Hist2d([80,50],[[-0.05,7.95],[-0.05,4.95]],'log2(GEN b pT)','RECO jet pT / 1st GEN b pT for matched jets','upplots/jetoverbpTvlogbpT1'),
        "jetoverbpTvlogbpT2":    Hist2d([80,50],[[-0.05,7.95],[-0.05,4.95]],'log2(GEN b pT)','RECO jet pT / 2nd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT2'),
        "jetoverbpTvlogbpT3":    Hist2d([80,50],[[-0.05,7.95],[-0.05,4.95]],'log2(GEN b pT)','RECO jet pT / 3rd GEN b pT for matched jets','upplots/jetoverbpTvlogbpT3'),
        "jetoverbpTvlogbpT4":    Hist2d([80,50],[[-0.05,7.95],[-0.05,4.95]],'log2(GEN b pT)','RECO jet pT / 4th GEN b pT for matched jets','upplots/jetoverbpTvlogbpT4'),
    }
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open our file and grab the events tree
        f = uproot.open(files[fnum])#'nobias.root')
        events = f.get('Events')

        pdgida = events.array('GenPart_pdgId')
        parida = events.array('GenPart_genPartIdxMother')

        bs = PhysObj('bs')

        bs.eta = pd.DataFrame(events.array('GenPart_eta')[abs(pdgida[parida])==Aid]).rename(columns=inc)
        bs.phi = pd.DataFrame(events.array('GenPart_phi')[abs(pdgida[parida])==Aid]).rename(columns=inc)
        bs.pt  = pd.DataFrame(events.array('GenPart_pt')[abs(pdgida[parida])==Aid]).rename(columns=inc)
        
        jets = PhysObj('jets')

        jets.eta= pd.DataFrame(events.array('Jet_eta')).rename(columns=inc)
        jets.phi= pd.DataFrame(events.array('Jet_phi')).rename(columns=inc)
        jets.pt = pd.DataFrame(events.array('Jet_pt')).rename(columns=inc)


        print('Processing ' + str(len(bs.eta)) + ' events')

        ## Figure out how many bs and jets there are
        nb = bs.eta.shape[1]
        njet= jets.eta.shape[1]

        ## Sort our b dataframes in descending order of pt
        temp_pt = pd.DataFrame()
        temp_eta = pd.DataFrame()
        temp_phi = pd.DataFrame()
        for i in range(1,nb+1):
            temp_pt[i] = bs.pt[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            temp_eta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            temp_phi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
        bs.pt = temp_pt
        bs.eta = temp_eta
        bs.phi = temp_phi
        del [temp_pt,temp_eta,temp_phi]

        ev = Event(bs.pt,bs,jets)
        jets.cut(jets.pt>0)
        bs.cut(bs.pt>0)
        ev.sync()

        ## Create our dR dataframe by populating its first column and naming it accordingly
        jbdr2 = pd.DataFrame(np.power(jets.eta[1]-bs.eta[1],2) + np.power(jets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
        ## Loop over jet x b combinations
        jbstr = [] 
        for j in range(1,njet+1):
            for b in range(1,nb+1):
                ## Make our column name
                jbstr.append("Jet "+str(j)+" b "+str(b))
                if (j+b==2):
                    continue
                ## Compute and store the dr of the given b and jet for every event at once
                jbdr2[jbstr[-1]] = pd.DataFrame(np.power(jets.eta[j]-bs.eta[b],2) + np.power(jets.phi[j]-bs.phi[b],2))

        ## Create a copy array to collapse in jets instead of bs
        blist = []
        for b in range(nb):
            blist.append(jbdr2.filter(like='b '+str(b+1)))
            blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
            blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))
        bjdr2 = pd.concat(blist,axis=1,sort=False)
        ## Replace all values but the lowest dRs with 0s
        #jbdr2 = jbdr2[jbdr2.rank(axis=1,method='first') == 1].fillna(0)
        bjdr2 = bjdr2.fillna(0)
        

        for i in range(4):
            plots['bdRvlogbpT'+str(i+1)].dfill(np.log2(bs.pt[[i+1]]),blist[i])

            yval = np.divide(jets.pt[blist[i]>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0],bs.pt[[i+1]].dropna().reset_index(drop=True)[i+1])
            xval = np.log2(bs.pt[[i+1]]).melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0]
            plots['jetoverbpTvlogbpT'+str(i+1)].fill(xval,yval)

            
            plots['GoverRjetpT'].dfill(np.divide(blist[i],jets.pt))

            bjplots['bpT'+str(i+1)].dfill(bs.pt[[i+1]])
            bjplots['beta'+str(i+1)].dfill(bs.eta[[i+1]])
            bjplots['bjetpT'+str(i+1)].dfill(jets.pt[blist[i]>0])
            bjplots['bjeteta'+str(i+1)].dfill(jets.eta[blist[i]>0])

        ## Fill a jet pt array populated only by the jet in each event with the lowest dR to any b
        plots['bdR'].dfill(np.sqrt(bjdr2[bjdr2!=0]))
        plots['bphi'].dfill(bs.phi)#.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        plots['RjetpT'].dfill(jets.pt)
    plt.clf()
    plots.pop('bdR').plot(logv=True)
    for p in plots:
        plt.clf()
        plots[p].plot()
    for p in bjplots:
        plt.clf()
        bjplots[p].plot()
    ## Draw the jet pT plot
    #plt.clf()
    #ptplot = plots['jetpT'].make()
    #plt.xlabel('Jet pT (for lowest dR to Muon)')
    #plt.ylabel('Events')
    #plt.savefig('upplots/JetdRpT.png')
    #plt.show()
    ### Draw the jet dR plot
    #plt.figure(2)
    #plt.clf()
    #drplot = plots['jetdR'].make(logv=True)
    #plt.xlabel('Lowest dR between any Jet and Muon')
    #plt.ylabel('Events')
    #plt.savefig('upplots/JetMudR.png')
    #plt.show()
    sys.exit()


def trig(files):
    ## Create a dictionary of histogram objects
    plots = {
        'hltplot':  Hist(80,(-0.5,79.5),'Highest Muon pT','Events','upplots/TrigHLTplot'),
        'ptplot':   Hist(80,(-0.5,79.5),'Highest Muon pT','Events','upplots/TrigpTplot'),
        'ratioplot':Hist(80,(-0.5,79.5),'Highest Muon pT','HLT_Mu7_IP4 / Events with Muons of sip > 5','upplots/TrigRatioPlot')
    }
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over all input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open the file and retrieve our key branches
        f = uproot.open(files[fnum])
        events = f.get('Events')
        Muon = PhysObj('Muon',files[fnum],'pt','eta','phi','sip3d')
        Trig = PhysObj('trig')
        Trig.vals = pd.DataFrame(events.array('HLT_Mu7_IP4_part0')).rename(columns=inc)
        ev = Event(Muon.pt,Muon,Trig)
        print('Processing ' + str(len(Muon.pt)) + ' events')

        ## Cut pT to only muons with SIP > 8
        Muon.cut(Muon.sip3d>5)
        #Trig.cut(Trig.vals)
        Trig.trim(Muon.pt)
        ## Create the two histograms we want to divide
        plt.figure(1)
        plots['ptplot'].fill(Muon.pt.max(axis=1))
        plots['hltplot'].fill(Muon.pt[Trig.vals].max(axis=1).dropna(how='all'))
    plots['ratioplot'].add(plots['hltplot'].divideby(plots['ptplot'],split=True))
    for pl in plots:
        plt.clf()
        plots[pl].plot()
    sys.exit()

def main():
    if (len(sys.argv) > 1): 
        files=[]
        ## Check for file sources
        if '-f' in sys.argv:
            idx = sys.argv.index('-f')+1
            for i in sys.argv[idx:]:
                files.append(i)
        elif '-l' in sys.argv:
            with open(sys.argv[3],'r') as rfile:
                for line in rfile:
                    files.append(line.strip('\n')) 
        else:
            files.append('NMSSM-20.root')
        ## Check specified run mode
        if sys.argv[1] == '-mc':
            mc(files)
        elif sys.argv[1] == '-jets':
            jets(files)
        elif sys.argv[1] == '-trig':
            trig(files)
        else:#if sys.argv[1] == '-a':
            ana(files)
 
    print("Expected n00dle.py <switch> (flag) (target)")
    print("-----switches-----")
    print("-mc    Runs a b-parent analysis on MC")
    print("-trig  Analyzes trigger efficiency for data")
    print("---optional flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)
    

## Define 'main' function as primary executable
if __name__ == '__main__':
    main()
