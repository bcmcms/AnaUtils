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

def inc(var):
    return var+1

def mc():
    f = uproot.open('GGH.root')
    events = f.get('Events')
    pdgid = events.array('GenPart_pdgId')
    parid = events.array('GenPart_genPartIdxMother')
    outlist = []
    for event in range(pdgid.size):
        for iGen in range(pdgid[event].size):
            if (abs(pdgid[event][iGen]) == 5) or (abs(pdgid[event][iGen] == 7)):
                parentIdx = parid[event][iGen]
                if parentIdx == -1: continue
                parentId = pdgid[event][parentIdx]
                if abs(parentId) == 9000006:
                    outlist.append(36)
                else:
                    outlist.append(parentId)
    plot = plt.hist(outlist,40,(-0.5,39.5),log=True)
    
    plt.xlabel('Parent PdgId')
    plt.ylabel('Number of b Children')
    plt.savefig('upplots/parents.png')
    #plt.show()
    plt.clf()
    #sys.exit()
    #return plot

def jets():
    ## Open our file and grab the events tree
    f = uproot.open('NMSSM-20.root')#'nobias.root')
    events = f.get('Events')
    ## Pick relevant branches into dataframes so we can work with them. We shift our indexes to avoid having a 0th jet.
    # This is a more compact form of
    # a = events.array('Branch')
    # b = pd.DataFrame(a)
    # c = b.rename(colums=inc)
    ## 
    mueta = pd.DataFrame(events.array('Muon_eta')).rename(columns=inc)
    muphi = pd.DataFrame(events.array('Muon_phi')).rename(columns=inc)
    jeteta= pd.DataFrame(events.array('Jet_eta')).rename(columns=inc)
    jetphi= pd.DataFrame(events.array('Jet_phi')).rename(columns=inc)
    jetpt = pd.DataFrame(events.array('Jet_pt')).rename(columns=inc)
    ## Figure out how many muons and jets there are
    nmu = mueta.shape[1]
    njet= jeteta.shape[1]
    ## Create our result dataframe by populating its first column and naming it accordingly
    jmudr2 = pd.DataFrame(np.power(jeteta[1]-mueta[1],2) + np.power(jetphi[1]-muphi[1],2)).rename(columns={1:'Jet 1 Mu 1'})
    ## Loop over jet x muon combinations
    jmustr= []
    for j in range(1,njet+1):
        for m in range(1,nmu+1):
            ## Make our column name
            jmustr.append("Jet "+str(j)+" Mu "+str(m))
            ## Skip the 1st column since that was used to initialize the dataframe
            if (j+m==2):
                continue
            ## Compute and store the dr of the given muon and jet for every event at once
            jmudr2[jmustr[-1]] = pd.DataFrame(np.power(jeteta[j]-mueta[m],2) + np.power(jetphi[j]-muphi[m],2))
    ## Replace all values but the lowest dRs with 0s
    jmudr2 = jmudr2[jmudr2.rank(axis=1,method='min') == 1].fillna(0)
    ## For every entry in the table
    for j in range(1,njet+1):
        head= "Jet "+str(j)+" Mu 1"
        for m in range(2,nmu+1):
            col = "Jet "+str(j)+" Mu "+str(m)
            ## Add Muons 2 through N to Muon 1
            jmudr2[head]=jmudr2[head]+jmudr2[col]
            ## Then remove Muons 2 through N
            jmudr2=jmudr2.drop(columns=[col]) 
        ## As we shape our dataframe to match the dimensions of jetpt, make the column names match
        jmudr2=jmudr2.rename(columns={head:j})
    ## Plot a jet pt array populated only by the jet in each event with the lowest dR to any muon
    plt.figure(1)
    pts = jetpt[jmudr2>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
    ptplot = plt.hist(pts[0],100,(0,100))
    plt.xlabel('Jet pT (for lowest dR to Muon)')
    plt.ylabel('Events')
    plt.savefig('upplots/JetdRpT.png')
    plt.clf()
    pts = jmudr2[jmudr2!=0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
    drplot = plt.hist(pts[0],100,(0,.1),log=True)
    plt.xlabel('Lowest dR between any Jet and Muon')
    plt.ylabel('Events')
    plt.savefig('upplots/JetMudR.png')
    #plt.show()
    plt.clf()
    #sys.exit()
    ## Isolate the lowest DR for each event, clean it by dumping it to a list, then cycle it back into a dataframe.
    # This is equivalent to writing
    # mask = [a.rank(axis=1, method='min) == 1]
    # b = a[mask]
    # c = b.stack() 
    # d = c.index
    # e = d.tolist()
    # f = pd.DataFrame(e)
    ##
    #jmudr2 = pd.DataFrame(jmudr2[jmudr2.rank(axis=1,method='min') == 1].stack().index.tolist())
    ## Clean up the dataframe into one event-compatible index and a string of where the lowest muon is.
    #jmudr2 = jmudr2.set_index(jmudr2[0]).drop(columns=[0])       
    #jets = jmudr2[1].str.split(n = 3, expand=True)[1]

def trig():
    f = uproot.open('Parked.root')#'nobias.root')
    events = f.get('Events')
    muon_pt = pd.DataFrame(events.array('Muon_pt'))
    muon_sip = pd.DataFrame(events.array('Muon_sip3d'))
    ## It's important to pick a trigger that was actually used for the data
    trig = pd.DataFrame(events.array('HLT_Mu9_IP5_part0'))
    ## Cut pT to only muons with SIP > 8
    muon_pt = muon_pt[muon_sip>8]
    ## Melt down the muon_pt array into a 1D list to plot
    pts = muon_pt.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
    ## Create the two histograms we want to divide
    plt.figure(1)
    ptplot = plt.hist(pts[0],100,(0,100))
    hltplot = plt.hist(pts[trig][0],100,(0,100))
    plt.clf()
    ## Pick out the bin values from the plotted histos and divide them
    ws=np.divide(hltplot[0],ptplot[0], where=ptplot[0]!=0)
    ## Empty bins should have a weight of 0
    ws[np.isnan(ws)] = 0
    ## Plot the histogram bins weighted with the result of the ratio
    ratioplot = plt.hist(ptplot[1][:100],100,(0,100),weights=ws)
    
    plt.xlabel('Muon pT')
    plt.ylabel('HLT_Mu9_IP5_part0 trigger frequency at SIP>8')
    plt.savefig('upplots/TrigRatio.png')
    #plt.show()
    plt.clf()
    #sys.exit()

def main():
    mc()
    jets()
    trig()
    sys.exit(0)
    

## Define 'main' function as primary executable
if __name__ == '__main__':
    main()
