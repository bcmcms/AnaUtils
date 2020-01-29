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

class hist(object):
    def __init__(s,size,bounds,xlabel='',ylabel='',fname=''):
        s.size = size
        s.bounds = bounds
        s.hs = [plt.hist([],size,bounds)[0],plt.hist([],size,bounds)[1]]
        s.xlabel = xlabel
        s.ylabel = ylabel
        s.fname = fname

    def __getitem__(s,i):
        if (i > 1) or (i < -2):
            raise Exception('histo object was accessed with an invalid index')
        return s.hs[i]

    ## Adds the values of a passed histogram to the class's plot
    def add(s,inplot):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        s.hs[0] = s.hs[0] + inplot[0]
        return s

    ## Fills the stored histogram with the supplied values
    def fill(s,vals):
        s.hs[0] = s.hs[0] + plt.hist(vals,s.size,s.bounds)[0]
        return s

    ## Fills the stored histogram with values from the supplied dataframe
    def dfill(s,frame):
        s.fill(frame.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        return s

    ## Divides the stored histogram by another, and either changes itself or returns a changed object
    def divideby(s,inplot,split=False):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        if split:
            s = cp.deepcopy(s)
        s.hs[0] = np.divide(s.hs[0],inplot[0], where=inplot[0]!=0)
        ## Empty bins should have a weight of 0
        s.hs[0][np.isnan(s.hs[0])] = 0
        return s

    ## Creates and returns a pyplot-compatible histogram object
    def make(s,logv=False):
        return plt.hist(s.hs[1][:s.size],s.size,s.bounds,weights=s.hs[0],log=logv)

    def plot(s,logv=False):
        s.make(logv)
        if s.xlabel != '':
            plt.xlabel(s.xlabel)
        if s.ylabel != '':
            plt.ylabel(s.ylabel)
        if s.fname != '':
            plt.savefig(s.fname)

def inc(var):
    return var+1

def mc(files):
    ## This histogram object is used to accumulate and render our data, defined above
    pdgplt = hist(40,(-0.5,39.5))
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
    ## Make a dictionary of histogram objects
    plots = {
        "beta":     hist(629,(-3.14,3.14),'GEN b Eta','Events','upplots/beta'),
        "bphi":     hist(629,(-3.14,3.14),'GEN b Phi','Events','upplots/bphi'),
        "bdR":      hist(100,(-0.0005,0.0995),'Lowest GEN b-jet dR','Events','upplots/bdR'),
        "bpT":    hist(100,(-0.5,99.5),'GEN pT of b','Events','upplots/bdRpT'),
        "bdRjetpT": hist(100,(-0.5,99.5),'RECO pT of jet with lowest b dR','Events','upplots/bdRjetpT'),
        "bdRgjetpT":hist(100,(-0.5,99.5),'GEN pT of jet with lowest b dR','Events','upplots/bdRgjetpT'),
        "GjetpT":   hist(100,(-0.5,99.5),'GEN jet pT','Events','upplots/GjetpT'),
        "RjetpT":   hist(100,(-0.5,99.5),'RECO jet pT','Events','upplots/RjetpT'),
        "GRjetpT":  hist(100,(-0.5,99.5),'jet pT','Ratio of GEN/RECO jet pT for jets with lowest dR to b','upplots/GRjetpT')
    }
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open our file and grab the events tree
        f = uproot.open(files[fnum])#'nobias.root')
        events = f.get('Events')
        ## Pick relevant branches into dataframes so we can work with them. We shift our indexes to avoid having a 0th jet.
        # This is a more compact form of
        # a = events.array('Branch')
        # b = pd.DataFrame(a)
        # c = b.rename(colums=inc)
        ##
        pdgida = events.array('GenPart_pdgId')
        parida = events.array('GenPart_genPartIdxMother')
        beta= pd.DataFrame(events.array('GenPart_eta')[abs(pdgida[parida])==9000006]).rename(columns=inc)
        bphi= pd.DataFrame(events.array('GenPart_phi')[abs(pdgida[parida])==9000006]).rename(columns=inc)
        bpt = pd.DataFrame(events.array('GenPart_pt')[abs(pdgida[parida])==9000006]).rename(columns=inc)
        jeteta= pd.DataFrame(events.array('Jet_eta')).rename(columns=inc)
        jetphi= pd.DataFrame(events.array('Jet_phi')).rename(columns=inc)
        jetpt = pd.DataFrame(events.array('Jet_pt')).rename(columns=inc)
        genjeteta= pd.DataFrame(events.array('GenJet_eta')).rename(columns=inc)
        genjetphi= pd.DataFrame(events.array('GenJet_phi')).rename(columns=inc)
        genjetpt = pd.DataFrame(events.array('GenJet_pt')).rename(columns=inc)
        print('Processing ' + str(len(beta)) + ' events')
        ## Figure out how many bs and jets there are
        nb = beta.shape[1]
        njet= jeteta.shape[1]
        ngjet = genjeteta.shape[1]

        ## Create our dR dataframe by populating its first column and naming it accordingly
        jbdr2 = pd.DataFrame(np.power(jeteta[1]-beta[1],2) + np.power(jetphi[1]-bphi[1],2)).rename(columns={1:'Jet 1 b 1'})
        genjbdr2 = pd.DataFrame(np.power(genjeteta[1]-beta[1],2) + np.power(genjetphi[1]-bphi[1],2)).rename(columns={1:'Jet 1 b 1'})

        ## Loop over jet x b combinations
        jbstr = [] 
        for j in range(1,njet+1):
            for b in range(1,nb+1):
                ## Make our column name
                jbstr.append("Jet "+str(j)+" b "+str(b))
                ## Skip the 1st column since that was used to initialize the dataframe
                if (j+b==2):
                    continue
                ## Compute and store the dr of the given b and jet for every event at once
                jbdr2[jbstr[-1]] = pd.DataFrame(np.power(jeteta[j]-beta[b],2) + np.power(jetphi[j]-bphi[b],2))
        ##Do the same for GEN jets
        genjbstr = []
        for j in range(1,ngjet+1):
            for b in range(1,nb+1):
                ## Make our column name
                genjbstr.append("Jet "+str(j)+" b "+str(b))
                ## Skip the 1st column since that was used to initialize the dataframe
                if (j+b==2):
                    continue
                ## Compute and store the dr of the given b and jet for every event at once
                genjbdr2[genjbstr[-1]] = pd.DataFrame(np.power(genjeteta[j]-beta[b],2) + np.power(genjetphi[j]-bphi[b],2))

        ## Create a copy array to collapse in jets instead of bs
        blist = []
        for b in range(nb):
            blist.append(jbdr2.filter(like='b '+str(b+1)))
            blist[b] = blist[b][blist[b].rank(axis=1,method='min') == 1]
        bjdr2 = pd.concat(blist,axis=1,sort=False)
        ## Replace all values but the lowest dRs with 0s
        jbdr2 = jbdr2[jbdr2.rank(axis=1,method='first') == 1].fillna(0)
        genjbdr2 = genjbdr2[genjbdr2.rank(axis=1,method='first') == 1].fillna(0)
        bjdr2 = bjdr2.fillna(0)

        ## For every entry in the table
        for j in range(1,njet+1):
            head= "Jet "+str(j)+" b 1"
            for b in range(2,nb+1):
                col = "Jet "+str(j)+" b "+str(b)
                ## Add bs 2 through N to b 1
                jbdr2[head]=jbdr2[head]+jbdr2[col]
                ## Then remove bs 2 through N
                jbdr2=jbdr2.drop(columns=[col])
            ## As we shape our dataframe to match the dimensions of jetpt, make the column names match
            jbdr2=jbdr2.rename(columns={head:j})
        ## For every entry in the table, but for GEN jets
        for j in range(1,ngjet+1):
            head= "Jet "+str(j)+" b 1"
            for b in range(2,nb+1):
                col = "Jet "+str(j)+" b "+str(b)
                ## Add bs 2 through N to b 1
                genjbdr2[head]=genjbdr2[head]+genjbdr2[col]
                ## Then remove bs 2 through N
                genjbdr2=genjbdr2.drop(columns=[col])
            ## As we shape our dataframe to match the dimensions of jetpt, make the column names match
            genjbdr2=genjbdr2.rename(columns={head:j})
        ## For every entry in the table, but the other direction
        for b in range(1,nb+1):
            head= "Jet 1 b "+str(b)
            for j in range(2,njet+1):
                col = "Jet "+str(j)+" b "+str(b)
                ## Add jets 2 through N to jet 1
                bjdr2[head]=bjdr2[head]+bjdr2[col]
                ## Then remove jets 2 through N
                bjdr2=bjdr2.drop(columns=[col])
            ## As we shape our dataframe to match the dimensions of bpt, make the column names match
            bjdr2=bjdr2.rename(columns={head:b})

        ## Fill a jet pt array populated only by the jet in each event with the lowest dR to any b
        #pts = jetpt[jbdr2>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        plots['bdRjetpT'].dfill(jetpt[jbdr2!=0])
        ## Fill an array with the dR between each b and its lowest dR jet in each event
        #pts = bjdr2[bjdr2!=0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        plots['bdR'].dfill(np.sqrt(bjdr2[bjdr2!=0]))
        ## Fill an array with the pt of bs in each event
        #pts = bpt[bjdr2>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        plots['bpT'].dfill(bpt)
        plots['beta'].dfill(beta)#.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        plots['bphi'].dfill(bphi)#.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        plots['bdR'].dfill(bjdr2[bjdr2!=0])#.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        plots['bdRgjetpT'].dfill(genjetpt[genjbdr2!=0])#.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        plots['GjetpT'].dfill(genjetpt)
        plots['RjetpT'].dfill(jetpt)
    plots['GRjetpT'].add(plots['bdRjetpT'].divideby(plots['bdRgjetpT'],split=True))
    for p in plots:
        plt.clf()
        plots[p].plot()
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


def jets(files):
    ## Make a dictionary of histogram objects
    plots = {
        "jetpT": hist(100,(-0.5,99.5)),
        "jetdR": hist(size=100,bounds=(-0.0005,0.0995))
    }
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open our file and grab the events tree
        f = uproot.open(files[fnum])#'nobias.root')
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
        print('Processing ' + str(len(mueta)) + ' events')
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
        ## Fill a jet pt array populated only by the jet in each event with the lowest dR to any muon
        pts = jetpt[jmudr2>0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        plots['jetpT'].fill(pts[0])
        pts = jmudr2[jmudr2!=0].melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        plots['jetdR'].fill(pts[0])
    ## Draw the jet pT plot
    plt.clf()
    ptplot = plots['jetpT'].make()
    plt.xlabel('Jet pT (for lowest dR to Muon)')
    plt.ylabel('Events')
    plt.savefig('upplots/JetdRpT.png')
    plt.show()
    ## Draw the jet dR plot
    plt.figure(2)
    plt.clf()
    drplot = plots['jetdR'].make(logv=True)
    plt.xlabel('Lowest dR between any Jet and Muon')
    plt.ylabel('Events')
    plt.savefig('upplots/JetMudR.png')
    plt.show()
    sys.exit()
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

def trig(files):
    ## Create a dictionary of histogram objects
    plots = {
        'hltplot':  hist(100,(-0.5,99.5)),
        'ptplot':   hist(100,(-0.5,99.5))
    }
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over all input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open the file and retrieve our key branches
        f = uproot.open(files[fnum])
        events = f.get('Events')
        muon_pt = pd.DataFrame(events.array('Muon_pt'))
        muon_sip = pd.DataFrame(events.array('Muon_sip3d'))
        print('Processing ' + str(len(muon_pt)) + ' events')
        ## It's important to pick a trigger that was actually used for the data and file in question
        trig = pd.DataFrame(events.array('HLT_Mu9_IP5_part0'))
        #trig2 = pd.DataFrame(events.array('HLT_Mu9_IP6_part0'))
        ## Cut pT to only muons with SIP > 8
        muon_pt = muon_pt[muon_sip>8]
        ## Melt down the muon_pt array into a 1D list to plot
        pts = muon_pt.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)
        ## Create the two histograms we want to divide
        plt.figure(1)
        plots['ptplot'].fill(pts[0])
        plots['hltplot'].fill(pts[trig][0])
    ## Pick out the bin values from the plotted histos and divide them
    ptplotf = plots['ptplot'].make()
    hltplotf = plots['hltplot'].make()
    ws=np.divide(hltplotf[0],ptplotf[0], where=ptplotf[0]!=0)
    ## Empty bins should have a weight of 0
    ws[np.isnan(ws)] = 0
    ## Plot the histogram bins weighted with the result of the ratio
    plt.clf()
    ratioplot = plt.hist(ptplotf[1][:100],100,(0,100),weights=ws)
    
    plt.xlabel('Muon pT')
    plt.ylabel('HLT_Mu9_IP5_part0 trigger frequency at SIP>8')
    plt.savefig('upplots/TrigRatio.png')
    plt.show()
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
    print("-jets  Analyzes jet/muon dR and pT")
    print("-trig  Analyzes trigger efficiency for data")
    print("---optional flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)
    

## Define 'main' function as primary executable
if __name__ == '__main__':
    main()
