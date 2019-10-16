#! /usr/bin/env python

import ROOT as R
R.gROOT.SetBatch(True)  ## Don't display histograms or canvases when drawn

import os
import subprocess
import sys

## User-defined constants
MAX_EVT = 0
#MAX_EVT = 10000  ## Maximum number of events to process
PRT_EVT = 1000   ## Print to screen every Nth event
VERBOSE = False  ## Print extra info about each event
F_PREFIX = True  ## Attach a prefix that denotes which file the plots came from. Only uses the 1st file.


## User-defined selection
SEL_ZEE = False ## Select only events containing Z --> ee decays

class histo:                    ## Define our histo object, to keep all these properties in one place
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

def main(batch=0):

###################
## Initialize files
###################

    ## Location of input files: current working directory + data/
    file_names = []
    in_dir = os.getcwd()+'/data/'
    if batch > 0:                   ## Handles batch processing
        file_names.append(sys.argv[batch])
    elif (len(sys.argv) > 1):       ## Allows for files to be given as arguments
        for i in range(len(sys.argv)-1):
            file_names.append(sys.argv[i+1])
    else:                           ## Expand for a system of hard-coded files
        file_names.append(in_dir+'ZH_to_4Tau_1_file.root')

    for in_file_name in file_names:
        if not '.root' in in_file_name: continue
        print 'Opening file: '+in_file_name

    ## Chain together trees from input files
    in_chains = []
    for i in range(len(file_names)):
        in_chains.append( R.TChain('Events') )
        in_chains[i].Add( file_names[i] )

    ## Set output directories (create if they do not exist)
    if not os.path.exists('plots/png/BasicNanoReader/'):
        os.makedirs('plots/png/BasicNanoReader/')
        
    out_file = R.TFile('plots/BasicNanoReader.root', 'recreate')
    png_dir  = 'plots/png/BasicNanoReader/'
    ## Attach a prefix based on the first filename to the plots
    if (F_PREFIX) or (batch < 0):
        png_dir = png_dir + file_names[0].split(".")[0] + "-"

    
#############
## Histograms
#############

    ## Histogram bins: [# of bins, minimum x, maximum x]
    mu_pt_bins = [200, 0, 201]

    ## Book 1D histograms
    ## Important to use '1D' instead of '1F' when dealing with large numbers of entries, and weighted events (higher precision)
    ## Organized into a dictionary for easy iteration
    plots = {
        "h_mu_pt": histo('h_mu_pt', 'Muon pT;pT (GeV);Events', mu_pt_bins),
        "h_mu_pt_eta": histo('h_mu_pt_eta', 'Muon pT (|eta| < 1.5);pT (GeV);Events', mu_pt_bins),
        "h_met": histo('h_met', 'MET;Missing Transverse Energy (Gev);Events; ', [200,0,200]),
        "h_cutflow_l": histo('h_cutflow_l', 'Total Muons // Non-Prompt Muons // Loose L1T // HLT', [5,-0.5,4.5]),
        "h_cutflow_t": histo('h_cutflow_t', 'Total Muons // Non-Prompt Muons // Tight L1T // HLT', [5,-0.5,4.5])
    }

#############
## Event loop
#############
                
    iEvt = -1
    nPass = 0
    partList = []
    for ch in in_chains:
        
        if iEvt > MAX_EVT and MAX_EVT > 0: break            ## Breaks early when we hit MAX_EVT events. Disabled for MAX<= 0
                
        for jEvt in range(ch.GetEntries()):
            iEvt += 1
            
            if iEvt > MAX_EVT and MAX_EVT > 0: break        ## Duplication of previous check - fires more often
            if iEvt % PRT_EVT is 0: print 'Event #', iEvt   ## Prints progress messages every PRT_EVT events

            ch.GetEntry(jEvt)

            nGen   = len(ch.GenPart_pdgId)  ## Number of GEN particles in the event
            nMuons = len(ch.Muon_pt)        ## Number of RECO muons in the event
            muPt = []
            muEta = []
            muPhi = []

            if VERBOSE: print '\nIn event %d we find %d GEN particles and %d RECO muons'  % (iEvt, nGen, nMuons)

            ## Fill pre-cut histos
            plots["h_met"].cfill(ch.MET_pt)
            plots["h_cutflow_l"].cfill(0)
            plots["h_cutflow_t"].cfill(0)

            ## Looking for 'A' (Spoilers: It's PDG ID = 36)
            #for iGen in range(nGen):
            #    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
            #    if parentIdx == -1: continue                    ## Skip particles with unknown parents
            #    if ch.GenPart_pdgId[parentIdx] == 25:           ## If a particle has a Higgs parent...
            #        partId = abs(ch.GenPart_pdgId[iGen])
            #        if partId in partList: continue             ## And hasn't appeared yet...
            #        partList.append(partId)                     ## Add it to the pile
            #        print(partList)

            ## Select only events containing Z-->ee decays
            #isZToEE = False
            #for iGen in range(nGen):
            #    if abs(ch.GenPart_pdgId[iGen]) != 11: continue  ## Only interested in electrons
            #    iGenMomIdx = ch.GenPart_genPartIdxMother[iGen]
            #    if iGenMomIdx == -1: continue  ## For particles with unknown parents
            #    iGenMomID  = ch.GenPart_pdgId[iGenMomIdx]
            #    if VERBOSE: print '  * iGen = %d, iGenMomIdx = %d, iGenMomID = %d' % (iGen, iGenMomIdx, iGenMomID)
            #    if iGenMomID == 23: ## Make sure the decaying particle is a Z
            #        isZToEE = True
            #        break           ## Found it - no need to keep looking
            #if SEL_ZEE and not isZToEE: continue
            #nPass +=1   ## increment the count of succesfull Z-->ee events 

            ## Tag muons from A, W, and Z decays for later plotting
            for iGen in range(nGen):
                if abs(ch.GenPart_pdgId[iGen]) == 13:               ## Particle is a muon
                    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
                    if parentIdx == -1: continue                    ## Skip particles with unknown parents
                    parentId = ch.GenPart_pdgId[parentIdx]          ## Get the parent PDG ID
                    if (parentId == 36) or (parentId == 24) or (parentId == 23):    ## If a particle has an A, W, or Z parent...
                        muEta.append(ch.GenPart_eta[iGen])
                        muPhi.append(ch.GenPart_phi[iGen])
                        muPt.append(ch.GenPart_pt[iGen])            ## Store its eta, phi, and pT

            ## Loop over RECO muons
            goodMu = []
            promptCut=False
            L1TCut=False
            L1TCutT=False
            HLTCut=False
            HLTCutT=False
            for iMu in range(nMuons):

                ## Get variable quantities out of the tree
                mu_pt = ch.Muon_pt[iMu]
                mu_eta = ch.Muon_eta[iMu]
                mu_phi = ch.Muon_phi[iMu]
                mu_sip = ch.Muon_sip3d[iMu]

                badMu = False
                ## Check if the muon is a match for any GEN muons from W, Z, or A decays
                for i in range(len(muEta)):
                    ## Check all our W, Z and A candidates to see if our muon matches one
                    if (abs(mu_eta - muEta[i]) < 0.1
                        ) and (abs(mu_pt - muPt[i]) < muPt[i]*.2
                        ) and ((abs(mu_phi - muPhi[i]) < 0.1) or (abs(mu_phi - muPhi[i]) > 6.23)):
                        badMu = True        ## This muon comes from something we're not interested in
                        break
                if badMu:
                    continue            ## Skip to the next muon if this one isn't wanted
                else:
                    promptCut = True    ## Otherwise, mark that the cut passed

                ## L1Trigger cut based on L1_SingleMu6er1p5 (loose)
                if (abs(mu_eta) > 1.5) or (abs(mu_pt) < 6.0):
                    badMu = True
                    continue      
                else:
                    L1TCut = True

                ## L1Trigger cut based on L1_vSingleMu18er1p5 (tight)
                if (abs(mu_eta) > 1.5) or (abs(mu_pt) < 18.0):
                    badMu = True
                    #continue           ## Failing a tight trigger can still allow loose trigger data to be collected
                else:
                    L1TCutT = True

                ## HLT cut based on HLT_Mu7_IP4 (loose)
                if (abs(mu_pt) < 7.0) or (mu_sip < 4):
                    badMu = True
                    continue
                else:
                    HLTCut = True

                ## HLT cut based on HLT_Mu12_IP6 (tight)
                if ((abs(mu_pt) < 12.0) or (mu_sip < 6)) and not badMu:
                    badMu = True
                    #continue
                elif not badMu:
                    HLTCutT = True

                goodMu.append(iMu)          ## This muon passes all of our cuts, so we save it

                if VERBOSE: print '  * Muon_pt[%d] = %.2f GeV' % (iMu, mu_pt)

                ## Fill the histograms
                plots["h_mu_pt"].cfill(mu_pt)
    
                ## Fill only muons with |eta| < 1.5
                if abs(ch.Muon_eta[iMu]) < 1.5:
                    plots["h_mu_pt_eta"].cfill(mu_pt)                

            ## Fill cut flow plots based on what was passed (order matters)
            if promptCut:
                plots["h_cutflow_l"].cfill(1) 
                plots["h_cutflow_t"].cfill(1)
            if L1TCut:
                plots["h_cutflow_l"].cfill(2)
            if L1TCutT:
                plots["h_cutflow_t"].cfill(2)
            if HLTCut:
                plots["h_cutflow_l"].cfill(3)
            if HLTCutT: 
                plots["h_cutflow_t"].cfill(3)

            ## End loop over RECO muons pairs (iMu)
            
        ## End loop over events in chain (jEvt)
    ## End loop over chains (ch)

    print '\nOut of %d total events processed, %d passed selection cuts' % (iEvt+1, nPass)

#######################
## Histogram formatting
#######################

    plots["h_cutflow_l"].h.Scale(1.0/plots["h_cutflow_l"].h.GetBinContent(1))
    plots["h_cutflow_l"].h.SetMarkerSize(1.8)
    plots["h_cutflow_t"].h.Scale(1.0/plots["h_cutflow_t"].h.GetBinContent(1))
    plots["h_cutflow_t"].h.SetMarkerSize(1.8)

######################
## Save the histograms
######################

    out_file.cd()

    canvY = R.TCanvas('canvY')
    canvY.cd()
    R.gPad.SetLogy()
    canv = R.TCanvas('canv')


    ## Example of plotting something that needs specific settings
    canv.cd()
    hist = plots.pop("h_cutflow_l")
    hist.saveplot(canv, png_dir, drawop="htext")
    hist = plots.pop("h_cutflow_t")
    hist.saveplot(canv, png_dir, drawop="htext")

    ## Generic plotting loop, for your histos that don't have anything special    
    canv.cd()
    for i in plots:
        plots[i].saveplot(canv, png_dir)

    del out_file


## Define 'main' function as primary executable
if __name__ == '__main__':

    ## This block handles batch processing mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "-b":
            for i in range(len(sys.argv) - 2):
                main(i+2)
            sys.exit()

    main()

