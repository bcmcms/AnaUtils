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
F_PREFIX = False ## Attach a prefix that denotes which file the plots came from. Only uses the 1st file.


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

    def markbins(s, markersize=1.8):
        s.h.Scale(1.0/s.h.GetBinContent(1))
        s.h.SetMarkerSize(markersize)


class trigger:
    def __init__(s, size):
        s.size = size
        s.list = [0]*size

    def store(s, index, val):
        if s.list[index] < val:
            s.list[index] = val

    def check(s, cutlevel, keepHigher=True):
        if keepHigher:
            return max(s.list[cutlevel:])
        else:
            return max(s.list[:cutlevel])

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
    f_prefix = file_names[0].split(".")[0]
    print(f_prefix)
    if not os.path.exists('plots/png/'+f_prefix+'/'):
        os.makedirs('plots/png/'+f_prefix+'/')
        
    out_file = R.TFile('plots/BasicNanoReader.root', 'recreate')
    png_dir  = 'plots/png/'+f_prefix+'/'
    ## Attach a prefix based on the first filename to the plots
    if (F_PREFIX): #or (batch < 0):
        png_dir = png_dir + f_prefix + "-"

    
#############
## Histograms
#############

    ## Histogram bins: [# of bins, minimum x, maximum x]
    mu_pt_bins = [200, 0, 201]
    indiv_trigger_bins = [3,-0.5,2.5]
    index_bins = [4,-0.5,3.5]

    ## Book 1D histograms
    ## Important to use '1D' instead of '1F' when dealing with large numbers of entries, and weighted events (higher precision)
    ## Organized into a dictionary for easy iteration
    plots = {
        "h_mu_pt": histo('h_mu_pt', 'Muon pT;pT (GeV);Events', mu_pt_bins),
        "h_mu_pt_eta": histo('h_mu_pt_eta', 'Muon pT (|eta| < 1.5);pT (GeV);Events', mu_pt_bins),
        "h_met": histo('h_met', 'MET;Missing Transverse Energy (Gev);Events; ', [200,0,200]),
        "h_cutflow_mc": histo('h_cutflow_mc', 'Total Events // Events with H->4b muons', [3,-0.5,2.5]),
    }
    ## Specifically contains plots related to triggers
    trigplots = {
        "Mu12_IP6":     histo('HLT_Mu12_IP6',   'HLT_Mu12_IP6',     indiv_trigger_bins),
        "Mu9_IP6":      histo('HLT_Mu9_IP6',    'HLT_Mu9_IP6',      indiv_trigger_bins),
        "Mu9_IP5":      histo('HLT_Mu9_IP5',    'HLT_Mu9_IP5',      indiv_trigger_bins),
        "Mu9_IP4":      histo('HLT_Mu9_IP4',    'HLT_Mu9_IP4',      indiv_trigger_bins),
        "Mu8_IP6":      histo('HLT_Mu8_IP6',    'HLT_Mu8_IP6',      indiv_trigger_bins),
        "Mu8_IP5":      histo('HLT_Mu8_IP5',    'HLT_Mu8_IP5',      indiv_trigger_bins),
        "Mu8_IP3":      histo('HLT_Mu8_IP3',    'HLT_Mu8_IP3',      indiv_trigger_bins),
        "Mu7_IP4":      histo('HLT_Mu7_IP4',    'HLT_Mu7_IP4',      indiv_trigger_bins),
        "Mu18_ER1p5":   histo('L1_SingleMu18_er1p56',   'L1_SingleMu18_er1p56',     indiv_trigger_bins),
        "Mu16_ER1p5":   histo('L1_SingleMu16_er1p56',   'L1_SingleMu16_er1p56',     indiv_trigger_bins),
        "Mu14_ER1p5":   histo('L1_SingleMu14_er1p56',   'L1_SingleMu14_er1p56',     indiv_trigger_bins),
        "Mu12_ER1p5":   histo('L1_SingleMu12_er1p56',   'L1_SingleMu12_er1p56',     indiv_trigger_bins),
        "Mu10_ER1p5":   histo('L1_SingleMu10_er1p56',   'L1_SingleMu10_er1p56',     indiv_trigger_bins),
        "Mu9_ER1p5":    histo('L1_SingleMu9_er1p56',    'L1_SingleMu9_er1p56',      indiv_trigger_bins),
        "Mu8_ER1p5":    histo('L1_SingleMu8_er1p56',    'L1_SingleMu8_er1p56',      indiv_trigger_bins),
        "Mu7_ER1p5":    histo('L1_SingleMu7_er1p56',    'L1_SingleMu7_er1p56',      indiv_trigger_bins),
        "HLT_cutflow":  histo('HLT_cutflow',    'Passed//Mu7_IP4//Mu8_IP3/5/6//Mu9_IP4/5/6//Mu12_IP6', [10,-0.5,9.5]),
        "L1T_cutflow":  histo('L1T_cutflow',    'Passed//MU7/8/9/10/12/14/16/18_er1p5', [10,-0.5,9.5])
    }
    indexplots = {
        #"Index2":       histo('Index2',     'Index 2 2.0e34+ZB+HLTPhysics',     index_bins),	
        "Index3":       histo('Index3',     'Index 3 1.7e34',                   index_bins),
        "Index4":       histo('Index4',     'Index 4 1.5e34',                   index_bins),
        "Index5":       histo('Index5',     'Index 5 1.3e34',                   index_bins),
        "Index6":       histo('Index6',     'Index 6 1.1e34',                   index_bins),
        "Index7":       histo('Index7',     'Index 7 9.0e33',                   index_bins)#,
        #"Index8":       histo('Index8',     'Index 8 6.0e33',                   index_bins),
        #"Index9":       histo('Index9',     'Index 9 1.7 to 0.6 e34 No Parking',index_bins),
        #"Index10":      histo('Index10',    'Index 10 2.0e34',                  index_bins),
        #"Index11":      histo('Index11',    'Index 11 900b',                    index_bins),
        #"Index12":      histo('Index12',    'Index 12 600b',                    index_bins),
        #"Index13":      histo('Index13',    'Index 13 3b',                      index_bins),
        #"Index14":      histo('Index14',    'Index 14 3b_2coll',                index_bins)
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
            plots["h_cutflow_mc"].cfill(0)

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
            HLTSIP = trigger(25)
            L1TETA = trigger(25)
            for iMu in range(nMuons):

                ## Get variable quantities out of the tree
                mu_pt = ch.Muon_pt[iMu]
                mu_eta = ch.Muon_eta[iMu]
                mu_phi = ch.Muon_phi[iMu]
                mu_sip = ch.Muon_sip3d[iMu]

                ## Cuts and Triggers
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

                if mu_pt >= 24:                     ## If our pt is too large for our trigger object,
                    HLTSIP.store(24,mu_sip)         ## Just fill the top bin
                    L1TETA.store(24,abs(mu_eta))
                else:
                    HLTSIP.store(int(mu_pt),mu_sip) ## Round each pT down and fill the SIP in that bin
                    L1TETA.store(int(mu_pt),mu_eta)


                goodMu.append(iMu)          ## This muon passes all of our cuts, so we save it

                if VERBOSE: print '  * Muon_pt[%d] = %.2f GeV' % (iMu, mu_pt)

                ## Fill the histograms
                plots["h_mu_pt"].cfill(mu_pt)
    
                ## Fill only muons with |eta| < 1.5
                if abs(ch.Muon_eta[iMu]) < 1.5:
                    plots["h_mu_pt_eta"].cfill(mu_pt)                

            ## Fill cut flow plots based on what was passed (order matters)
            if promptCut:                           ## This signifies that a muon of interest was even in this event
                plots["h_cutflow_mc"].cfill(1) 
                trigplots["HLT_cutflow"].cfill(0)
                trigplots["L1T_cutflow"].cfill(0)
                for plot in indexplots:
                    indexplots[plot].cfill(0)

            stillMu = True
            cutplace = 1
            L1Tpt = [7, 8, 9, 10, 12, 14, 16, 18]           ## These are all the pt cuts for the L1 Trigger
            for i in L1Tpt:                             
                trigplots["Mu"+str(i)+"_ER1p5"].cfill(0)    ## Every event goes in the 0 bins
                if (L1TETA.check(i) > 1.5) and stillMu:     ## If there's still a fitting muon at or above this pt,
                    trigplots["Mu"+str(i)+"_ER1p5"].cfill(1)## Fill the appropriate plot with an event
                    trigplots["L1T_cutflow"].cfill(cutplace)## And fill the appropriate cutflow bin
                    cutplace = cutplace + 1
                else:
                    stillMu = False                     ## Once we run out of muons, stop looking

            cutplace = 1
            HLTpt = [7,8,8,8,9,9,9,12]                      ## Combined, these arrays cover our HLT values
            HLTip = [4,3,5,6,4,5,6,6]

            for i in range(len(HLTpt)):                     ## Loop over every HLT
                trigplots["Mu"+str(HLTpt[i])+"_IP"+str(HLTip[i])].cfill(0)
                if HLTSIP.check(HLTpt[i]) >= HLTip[i]:      ## If there's a fitting muon at or above this pt, fill
                    trigplots["Mu"+str(HLTpt[i])+"_IP"+str(HLTip[i])].cfill(1)
                    trigplots["HLT_cutflow"].cfill(cutplace)
                cutplace = cutplace + 1

            ##Replicate individual prescale indeces

            if L1TETA.check(12) > 1.5:
                indexplots["Index3"].cfill(1)
                if HLTSIP.check(12) > 6:
                    indexplots["Index3"].cfill(2)

            if L1TETA.check(10) > 1.5:
                indexplots["Index4"].cfill(1)
                if HLTSIP.check(9) > 5:
                    indexplots["Index4"].cfill(2)

            if L1TETA.check(9) > 1.5:
                indexplots["Index5"].cfill(1)
                if HLTSIP.check(8) > 5:
                    indexplots["Index5"].cfill(2)

            if L1TETA.check(8) > 1.5:
                indexplots["Index6"].cfill(1)
                if HLTSIP.check(7) > 4:
                    indexplots["Index6"].cfill(2)

            if L1TETA.check(7) > 1.5:
                indexplots["Index7"].cfill(1)
                if HLTSIP.check(7) > 4:
                    indexplots["Index7"].cfill(2)

            ## End loop over RECO muons pairs (iMu)
            
        ## End loop over events in chain (jEvt)
    ## End loop over chains (ch)

    print '\nOut of %d total events processed, %d passed selection cuts' % (iEvt+1, nPass)

#######################
## Histogram formatting
#######################

    plots["h_cutflow_mc"].markbins()

    for i in trigplots:
        trigplots[i].markbins()
    for i in indexplots:
        indexplots[i].markbins()

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
    for i in trigplots:
        trigplots[i].saveplot(canv, png_dir, drawop="htext")
    for i in indexplots:
        indexplots[i].saveplot(canv, png_dir, drawop="htext")

    ## Generic plotting loop, for your histos that don't have anything special    
    canv.cd()
    hist = plots.pop("h_cutflow_mc")
    hist.saveplot(canv, png_dir, drawop="htext")
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

