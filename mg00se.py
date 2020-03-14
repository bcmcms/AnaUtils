#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mg00se.py                               ###
###                                                                  ###
### Run without arguments for a list of options and examples         ###
########################################################################

import ROOT as R
R.gROOT.SetBatch(True)  ## Don't display histograms or canvases when drawn

import os
#import subprocess
import sys
from math import sqrt, pow, log2
from analib import Hist, Hist2d
import matplotlib.pyplot as plt

## User-defined constants
MAX_EVT = 0
#MAX_EVT = 10000  ## Maximum number of events to process
PRT_EVT = 1000   ## Print to screen every Nth event
VERBOSE = False  ## Print extra info about each event
F_PREFIX = False ## Attach a prefix that denotes which file the plots came from. Only uses the 1st file




class rhist:                    ## Define our histo object, to keep all these properties in one place
    def __init__(s, name, title, bounds):
        s.name = name           ## What the file will be named
        s.title = title         ## Can be in the form title;x axis;y axis
        s.bounds = bounds       ## Expects [number of bins, lowest bin, highest bin]
        s.h = R.TH1D(name, title, bounds[0], bounds[1], bounds[2]) ## h is short for histogram. It's easy to forget, but it's likely the most referenced property, so conciseness is important! 

    def cfill(s, var, weight=1):          ## Shorthand to allow for cleaner binning with no worry of overflow bins
        s.h.Fill(min(max(var, s.bounds[1]+0.01), s.bounds[2]-0.01), weight)

    def saveplot(s, canvas, savedir, drawop='', color=R.kBlue, width=2, suffix='.png'): ## Takes care of saving the plot
        s.h.SetLineWidth(width)
        s.h.SetLineColor(color)
        s.h.Write()
        s.h.Draw(drawop)
        canvas.SaveAs(savedir+s.name+suffix)

    def markbins(s, markersize=1.8, refbin=1):
        s.h.Scale(1.0/s.h.GetBinContent(refbin))
        s.h.SetMarkerSize(markersize)


class trigger:
    def __init__(s, size, polarity=1, failval=None):
        s.size = size
        s.list = [None]*size
        s.p = polarity/abs(polarity)
        s.failv = failval

    def store(s, index, val, maxval=True):
        if s.list[index] < (val*s.p):
            s.list[index] = (val*s.p)

    def check(s, cutlevel, maxval=True):
        val = max(s.list[cutlevel:])
        if val == None:
            return s.failv
        else: return val*s.p

def main(batch=0):

###################
## Initialize files
###################

    ## Location of input files: current working directory + data/
    file_names = []
    #in_dir = os.getcwd()+'/data/'
    DATA = False
    f_prefix = False
    if batch > 0:                   ## Handles batch processing
        file_names.append(sys.argv[batch])
    elif (len(sys.argv) > 1):                               ## Allows for files to be given as arguments
        ## Takes a file that is in turn a list of files
        if sys.argv[1] == '-f':
            with open(sys.argv[2],'r') as rfile:
                for line in rfile:
                    file_names.append(line.strip('\n'))
            if len(sys.argv) > 2: f_prefix = sys.argv[3]    ## Tries to get the new file prefix from the user
            else: f_prefix = 'xrootd'                       ## Or just sets it to a default
        ## Uses the supplied arguments as filenames
        else:
            for i in range(len(sys.argv)-1):                
                if sys.argv[i+1] == "-d":
                    DATA = True
                else: file_names.append(sys.argv[i+1])
    else:                           ## Expand for a system of hard-coded files
        #file_names.append(in_dir+'ZH_to_4Tau_1_file.root')
        ##for now, just make it a debug option that explains proper formatting.
        print('Expected: mg00se.py <switch> <files>')
        print('--- switches ---')
        print('-b - enables batch processing (each file is treated separately)')
        print('-d - enables data mode (no gen analysis is run for any events)')
        print('-f - switches to text file input (attempts to open the file and use each line as an xrootd location)')
        print('     if used, a third argument may be provided to set the output folder and plot prefix name')
        sys.exit(0)

    for in_file_name in file_names:
        if not '.root' in in_file_name: continue
        print('Opening file: '+in_file_name)

    ## Chain together trees from input files
    in_chains = []
    for i in range(len(file_names)):
        in_chains.append( R.TChain('Events') )
        in_chains[i].Add( file_names[i] )

    ## Set output directories (create if they do not exist)
    if not f_prefix: f_prefix = file_names[0].split(".")[0]
    print(f_prefix)
    if not os.path.exists('plots/png/'+f_prefix+'/'):
        os.makedirs('plots/png/'+f_prefix+'/')
        
    #out_file = R.TFile('plots/BasicNanoReader.root', 'recreate')
    png_dir  = 'plots/png/'+f_prefix+'/'
    ## Attach a prefix based on the first filename to the plots
    if (F_PREFIX): #or (batch < 0):
        png_dir = png_dir + f_prefix + "-"

    
#############
## Histograms
#############

    ## Histogram bins: [# of bins, minimum x, maximum x]
    #mu_pt_bins = [200, 0, 201]

    ## Book 1D histograms
    ## Important to use '1D' instead of '1F' when dealing with large numbers of entries, and weighted events (higher precision)
    ## Organized into a dictionary for easy iteration
    #plots = {
    #    "h_mu_pt": histo('h_mu_pt', 'Muon pT;pT (GeV);Events', mu_pt_bins),
    #} 
    plots = {
        "HpT":      Hist(60 ,(0,320)    ,'GEN Higgs pT','Events','mg00se/HpT'),
        "A1pT":     Hist(80 ,(0,160)    ,'Highest GEN A pT','Events','mg00se/A1pT'),
        "A2pT":     Hist(80 ,(0,160)    ,'Lowest GEN A pT','Events','mg00se/A2pT'),
        "AdR":      Hist(50 ,(0,5)      ,'GEN A1 to A2 dR','Events','mg00se/AdR'),
        "bdRA1":    Hist(50 ,(0,5)      ,'GEN dR between highest pT A child bs','Events','mg00se/bdRA1'),
        "bdRA2":    Hist(50 ,(0,5)      ,'GEN dR between lowest pT A child bs','Events','mg00se/bdRA2'),
        "bdetaA1":  Hist(34 ,(0,3.4)    ,'GEN |deta| between highest-A child bs','Events','mg00se/bdetaA1'),
        "bdetaA2":  Hist(34 ,(0,3.4)    ,'GEN |deta| between lowest-A child bs','Events','mg00se/bdetaA2'),
        "bdphiA1":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between highest-A child bs','Events','mg00se/bdphiA1'),
        "bdphiA2":  Hist(34 ,(0,3.4)    ,'GEN |dphi| between lowest-A child bs','Events','mg00se/bdphiA2'),
        "bphi":     Hist(66 ,(-3.3,3.3) ,'GEN b Phi','Events','mg00se/bphi'),
        "bjdR":     Hist(100,(0,2)      ,'All GEN bs to matched jet dR','Events','mg00se/bjdR'),
        "RjetpT":   Hist(100,(0,100)    ,'RECO matched jet pT','Events','mg00se/RjetpT'),
        "Rjeteta":  Hist(66 ,(-3.3,3.3) ,'RECO matched jet eta','Events','mg00se/Rjeteta'),
        #"RjetCSVV2":Hist(140 ,(-12,2)    ,'RECO matched jet btagCSVV2 score','events','mg00se/RjetCSVV2'),
        #"RjetDeepB":Hist(40 ,(-2.5,1.5) ,'RECO matched jet btagDeepB score','events','mg00se/RjetDeepB'),
        #"RjetDeepFB"    :Hist(24 ,(0,1.2)    ,'RECO matched jet btagDeepFlavB score','events','mg00se/RjetDeepFB'),
        "RA1pT":    Hist(80 ,(0,160)    ,'pT of RECO A1 objects constructed from matched jets','Events','mg00se/RA1pT'),
        "RA2pT":    Hist(80 ,(0,160)    ,'pT of RECO A2 objects constructed from matched jets','Events','mg00se/RA2pT'),
        "RA1mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A1 objects from matched jets','Events','mg00se/RA1mass'),
        "RA2mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A2 objects from matched jets','Events','mg00se/RA2mass'),
        "RA1dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A1 object','Events','mg00se/RA1dR'),
        "RA2dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A2 object','Events','mg00se/RA2dR'),
        "RA1deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A1 object','Events','mg00se/RA1deta'),
        "RA2deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A2 object','Events','mg00se/RA2deta'),
        "RA1dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A1 object','Events','mg00se/RA1dphi'),
        "RA2dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A2 object','Events','mg00se/RA2dphi'),
        "RHmass":   Hist(80 ,(0,160)     ,'reconstructed mass of Higgs object from reconstructed As','Events','mg00se/RHmass'),
        "RHpT":     Hist(100,(0,200)    ,'pT of reconstructed higgs object from reconstructed As','Events','mg00se/RHpT'),
        "RHdR":     Hist(50 ,(0,5)      ,'dR between A children of reconstructed higgs object','Events','mg00se/RHdR'),
        "RHdeta":   Hist(33 ,(0,3.3)    ,'|deta| between A children of reconstructed higgs object','Events','mg00se/RHdeta'),
        "RHdphi":   Hist(33 ,(0,3.3)    ,'|dphi| between A children of reconstructed higgs object','Events','mg00se/RHdphi'),
        ##
        "RalljetpT":    Hist(100,(0,100),'All RECO jet pT','Events','mg00se/RalljetpT'),
        "bjdRvlogbpT1":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 1st pT GEN b to matched RECO jet','mg00se/bjdRvlogbpT1'),
        "bjdRvlogbpT2":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 2nd pT GEN b to matched RECO jet','mg00se/bjdRvlogbpT2'),
        "bjdRvlogbpT3":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 3rd pT GEN b to matched RECO jet','mg00se/bjdRvlogbpT3'),
        "bjdRvlogbpT4":   Hist2d([80,200],[[0,8],[0,2]],'log2(GEN b pT)','dR from 4th pT GEN b to matched RECO jet','mg00se/bjdRvlogbpT4'),
        "jetoverbpTvlogbpT1":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 1st GEN b pT for matched jets','mg00se/jetoverbpTvlogbpT1'),
        "jetoverbpTvlogbpT2":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 2nd GEN b pT for matched jets','mg00se/jetoverbpTvlogbpT2'),
        "jetoverbpTvlogbpT3":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 3rd GEN b pT for matched jets','mg00se/jetoverbpTvlogbpT3'),
        "jetoverbpTvlogbpT4":    Hist2d([60,40],[[2,8],[0,4]],'log2(GEN b pT)','RECO jet pT / 4th GEN b pT for matched jets','mg00se/jetoverbpTvlogbpT4'),
    }

#############
## Event loop
#############
                
    iEvt = -1
    nPass = 0
    for ch in in_chains:
        
        if iEvt > MAX_EVT and MAX_EVT > 0: break            ## Breaks early when we hit MAX_EVT events. Disabled for MAX<= 0
                
        for jEvt in range(ch.GetEntries()):
            iEvt += 1
            
            if iEvt > MAX_EVT and MAX_EVT > 0: break        ## Duplication of previous check - fires more often
            if iEvt % PRT_EVT is 0: print('Event # '+str(iEvt))   ## Prints progress messages every PRT_EVT events

            ch.GetEntry(jEvt)

            if not DATA: 
                nGen   = len(ch.GenPart_pdgId)  ## Number of GEN particles in the event
            else: nGen = 0
            nMuons = len(ch.Muon_pt)            ## Number of RECO muons in the event
            #nJets  = len(ch.Jet_pt)
            bEta, bPhi, bPt, AEta, APhi, APt, hEta, hPhi, hPt = [],[],[],[],[],[],[],[],[]

            if VERBOSE: print('\nIn event '+str(iEvt)+' we find '+str(nGen)+' GEN particles and '+str(nMuons)+' RECO muons')

            ## Fill pre-cut histos


            ## identify bs, As, and higgs
            for iGen in range(nGen):
                if (abs(ch.GenPart_pdgId[iGen]) == 5):              ## If the particle is a b
                    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
                    if parentIdx == -1: continue                    ## Skip particles with unknown parents
                    parentId = ch.GenPart_pdgId[parentIdx]          ## Get the parent PDG ID
                    if (abs(parentId) == 36) or (abs(parentId) == 9000006): ## If the particle has an A parent:
                        bEta.append(ch.GenPart_eta[iGen])
                        bPhi.append(ch.GenPart_phi[iGen])
                        bPt.append(ch.GenPart_pt[iGen])            ## Store its eta, phi, and pT

                if (abs(ch.GenPart_pdgId[iGen]) == 36) or (abs(ch.GenPart_pdgId[iGen]) == 9000006): ## If the particle is an A
                    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
                    if parentIdx == -1: continue                    ## Skip particles with unknown parents
                    parentId = ch.GenPart_pdgId[parentIdx]          ## Get the parent PDG ID
                    if (abs(parentId) == 25):                       ## If the particle has a higgs parent:
                        AEta.append(ch.GenPart_eta[iGen])
                        APhi.append(ch.GenPart_phi[iGen])
                        APt.append(ch.GenPart_pt[iGen])            ## Store its eta, phi, and pT

                if (abs(ch.GenPart_pdgId[iGen]) == 25):             ## If the particle is a higgs
                    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
                    if parentIdx == -1: continue                    ## Skip particles with unknown parents
                    parentId = ch.GenPart_pdgId[parentIdx]          ## Get the parent PDG ID
                    if (abs(parentId) != 25):                       ## If the particle isn't a self-interaction:
                        hEta.append(ch.GenPart_eta[iGen])
                        hPhi.append(ch.GenPart_phi[iGen])
                        hPt.append(ch.GenPart_pt[iGen])            ## Store its eta, phi, and pT

            nbs = len(bPt)
            if nbs != 4:
                continue
            nAs = len(APt)
            if nAs != 2:
                continue
            nhs = len(hPt)
            if nhs != 1:
                continue
                       
            ## Re order As and bs by A pt
            if APt[1] > APt[0]:
                APt  = [APt[1],APt[0]]
                APhi = [APhi[1],APhi[0]]
                AEta = [AEta[1],AEta[0]]
                bPt  = [bPt[2],bPt[3],bPt[0],bPt[1]]
                bEta = [bEta[2],bEta[3],bEta[0],bEta[1]]
                bPhi = [bPhi[2],bPhi[3],bPhi[0],bPhi[1]]
 
            ## Produce sorted copy of bs
            tbPt = bPt
            sbPt, sbEta, sbPhi = [],[],[]
            for j in range(nbs):
                i = tbPt.index(max(tbPt))
                sbPt.append( bPt[i] )
                sbEta.append(bEta[i])
                sbPhi.append(bPhi[i])
                tbPt[i] = -10
            del tbPt

            bjetEta, bjetPhi, bjetPt, bjetDR, bjetMass, bjetIdx = [0]*nbs, [0]*nbs, [0]*nbs, [9.0]*nbs, [0]*nbs, [0]*nbs
            sbjetEta, sbjetPhi, sbjetPt, sbjetDR, sbjetMass, sbjetIdx = [0]*nbs, [0]*nbs, [0]*nbs, [9.0]*nbs, [0]*nbs, [0]*nbs
            jetPt = []
            ## Begin loop over jets
            for iJet in range(len(ch.Jet_pt)):
                jet_eta = abs(ch.Jet_eta[iJet])
                jet_phi = ch.Jet_phi[iJet]
                jet_pt = ch.Jet_pt[iJet]
                jet_mass = ch.Jet_mass[iJet]
                jetPt.append(jet_pt)
                ## Get dRs
                for i in range(nbs):
                    dr = sqrt(pow(bEta[i] - jet_eta, 2) + pow(bPhi[i] - jet_phi, 2))
                    if dr < bjetDR[i]:
                        bjetEta[i], bjetPhi[i], bjetPt[i], bjetDR[i], bjetMass[i], bjetIdx[i] = jet_eta, jet_phi, jet_pt, dr, jet_mass, iJet
                    sdr = sqrt(pow(sbEta[i] - jet_eta, 2) + pow(sbPhi[i] - jet_phi, 2))
                    if dr < sbjetDR[i]:
                        sbjetEta[i], sbjetPhi[i], sbjetPt[i], sbjetDR[i], sbjetMass, sbjetIdx[i] = jet_eta, jet_phi, jet_pt, sdr, jet_mass, iJet

            ## Skip the event if the number of unique resolved jets is less than the number of bs
            if 0 in bjetEta:
                continue
            if len(bjetIdx) != len(set(bjetIdx)):
                continue
            ## Cut events with matched jet dR > 0.4
            jetdRcut = False
            for dr in bjetDR:
                if dr > 0.4:
                    jetdRcut = True
            if jetdRcut:
                continue
            nPass+=1
            
            Tjets = []
            for i in range(nbs):
                tmp = R.TLorentzVector()
                tmp.SetPtEtaPhiM(bjetPt[i],bjetEta[i],bjetPhi[i],bjetMass[i])
                Tjets.append(tmp)
            TAs = [Tjets[0]+Tjets[1], Tjets[2]+Tjets[3]]
            Th  = [TAs[0]+TAs[1]]


            #for i in range(4):
                #print(log2(sbPt[i]), (sbjetPt[i] / sbPt[i]))
                #plots["jetoverbpTvlogbpT"+str(i+1)].fill(log2(sbPt[i]), (sbjetPt[i] / sbPt[i]))
                #plots["bjdRvlogbpT"+str(i+1)      ].fill(log2(sbPt[i]), sbjetDR[i])
            plots["RalljetpT"].fill(jetPt)
            plots["HpT"].fill(hPt[0])
            plots["AdR"].fill(sqrt(pow(AEta[0] - AEta[1], 2)+pow(APhi[0] - APhi[1], 2)))
            for i in range(2):
                plots["A"+str(i+1)+"pT"].fill(APt[i])
                plots["bdRA"+str(i+1)  ].fill(sqrt(pow(bEta[(2*i)] - bEta[(2*i)+1], 2) + pow(bPhi[(2*i)] - bPhi[(2*i)+1], 2)))
                plots["bdetaA"+str(i+1)].fill(abs(bEta[(2*i)] - bEta[(2*i)+1]))
                plots["bdphiA"+str(i+1)].fill(abs(bEta[(2*i)] - bEta[(2*i)+1]))
                plots["RA"+str(i+1)+"pT"  ].fill(TAs[i].Pt())
                plots["RA"+str(i+1)+"mass"].fill(TAs[i].M())
                plots["RA"+str(i+1)+"dR"  ].fill(sqrt(pow(Tjets[i*2].Eta() - Tjets[(i*2)+1].Eta(),2) + pow(Tjets[i*2].Phi() - Tjets[(i*2)+1].Phi(),2)))
                plots["RA"+str(i+1)+"deta"].fill(abs(Tjets[i*2].Eta() - Tjets[(i*2)+1].Eta()))
                plots["RA"+str(i+1)+"dphi"].fill(abs(Tjets[i*2].Phi() - Tjets[(i*2)+1].Phi()))
            plots["RHpT"].fill(Th[0].Pt())
            plots["RHdR"].fill(sqrt(pow(TAs[0].Eta() - TAs[1].Eta(),2) + pow(TAs[0].Phi() - TAs[1].Phi(),2)))
            plots["RHdeta"].fill(abs(TAs[0].Eta() - TAs[1].Eta()))
            plots["RHdphi"].fill(abs(TAs[0].Phi() - TAs[1].Phi()))
            plots["bphi"].fill(bPhi)
            plots["bjdR"].fill(bjetDR)
            plots["RjetpT" ].fill(bjetPt)
            plots["Rjeteta"].fill(bjetEta)


        ## End loop over events in chain (jEvt)
    ## End loop over chains (ch)

    print('\nOut of '+str(iEvt+1)+' total events processed, '+str(nPass)+' passed selection cuts')

#######################
## Histogram formatting
#######################



######################
## Save the histograms
######################

    plt.clf()
    plots.pop('bjdR').plot(logv=True)
    for p in plots:
        plt.clf()
        plots[p].plot()

## Define 'main' function as primary executable
if __name__ == '__main__':

    ## This block handles batch processing mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "-b":
            for i in range(len(sys.argv) - 2):
                main(i+2)
            sys.exit()

    main()

