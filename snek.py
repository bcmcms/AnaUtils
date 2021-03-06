#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility snek.py                                 ###
###                                                                  ###
### Run without arguments for a list of options and examples         ###
########################################################################

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
F_PREFIX = False ## Attach a prefix that denotes which file the plots came from. Only uses the 1st file

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
    in_dir = os.getcwd()+'/data/'
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
        print 'Expected: snek.py <switch> <files>'
        print '--- switches ---'
        print '-b - enables batch processing (each file is treated separately)'
        print '-d - enables data mode (no gen analysis is run for any events)'
        print '-f - switches to text file input (attempts to open the file and use each line as an xrootd location)'
        print '     if used, a third argument may be provided to set the output folder and plot prefix name'
        sys.exit(0)

    for in_file_name in file_names:
        if not '.root' in in_file_name: continue
        print 'Opening file: '+in_file_name

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
    jet_trigger_bins = [9,-2.5,6.5]
    jet_pt_bins = [200, -0.5,199.5]
    index_bins = [5,-0.5,4.5]

    ## Book 1D histograms
    ## Important to use '1D' instead of '1F' when dealing with large numbers of entries, and weighted events (higher precision)
    ## Organized into a dictionary for easy iteration
    plots = {
        "h_mu_pt": histo('h_mu_pt', 'Muon pT;pT (GeV);Events', mu_pt_bins),
        "h_mu_pt_eta": histo('h_mu_pt_eta', 'Muon pT (|eta| < 1.5);pT (GeV);Events', mu_pt_bins),
        "h_met": histo('h_met', 'MET;Missing Transverse Energy (Gev);Events; ', [200,0,200]),
        "h_cutflow_mc": histo('h_cutflow_mc', 'Total Events // Events with H->4b muons', [3,-0.5,2.5]),
        "Jet_pt":       histo('Jet_pt',         'Jet pT - |eta| < 2.5 - max 4/event', [150,-1,299]),
        "Jet_eta":      histo('Jet_eta',        'Jet |eta| - all pT - max 4/event',   [25,-0.05,2.45]),
        "Jet1_pt":      histo('Jet1_pt',        'Highest jet pT',       jet_pt_bins),
        "Jet2_pt":      histo('Jet2_pt',        '2nd Highest jet pT',   jet_pt_bins),
        "Jet3_pt":      histo('Jet3_pt',        '3rd Highest jet pT',   jet_pt_bins),
        "Jet4_pt":      histo('Jet4_pt',        '4th Highest jet pT',   jet_pt_bins),
        "Jet5_pt":      histo('Jet5_pt',        '5th Highest jet pT',   jet_pt_bins),
        "Jet6_pt":      histo('Jet6_pt',        '6th Highest jet pT',   jet_pt_bins),
        "Njet_pt30":    histo('Njet_pt30',      'All Events//Passed//0/1/2/3/4/5/6+ jets over 30GeV', jet_trigger_bins),
        "Njet_pt25":    histo('Njet_pt25',      'All Events//Passed//0/1/2/3/4/5/6+ jets over 25GeV', jet_trigger_bins),
        "Njet_pt20":    histo('Njet_pt20',      'All Events//Passed//0/1/2/3/4/5/6+ jets over 20GeV', jet_trigger_bins),
        "Njet_pt15":    histo('Njet_pt15',      'All Events//Passed//0/1/2/3/4/5/6+ jets over 15GeV', jet_trigger_bins)
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
        "HLT_cutflow":  histo('HLT_cutflow',    'All Events//Passed//Mu7_IP4//Mu8_IP3/5/6//Mu9_IP4/5/6//Mu12_IP6', [11,-0.5,10.5]),
        "L1T_cutflow":  histo('L1T_cutflow',    'All Events//Passed//MU7/8/9/10/12/14/16/18_er1p5', [11,-0.5,10.5])
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

            if not DATA: 
                nGen   = len(ch.GenPart_pdgId)  ## Number of GEN particles in the event
            else: nGen = 0
            nMuons = len(ch.Muon_pt)            ## Number of RECO muons in the event
            muPt = []
            muEta = []
            muPhi = []

            if VERBOSE: print '\nIn event %d we find %d GEN particles and %d RECO muons'  % (iEvt, nGen, nMuons)

            ## Fill pre-cut histos
            plots["h_met"].cfill(ch.MET_pt)
            plots["h_cutflow_mc"].cfill(0)
            for plot in trigplots:
                trigplots[plot].cfill(0)
            for plot in indexplots:
                indexplots[plot].cfill(0)
            for i in ['Njet_pt15','Njet_pt20','Njet_pt25','Njet_pt30']:
                plots[i].cfill(-2)

            ## Looking for 'A' (Spoilers: It's PDG ID = 36)
            #for iGen in range(nGen):
            #    parentIdx = ch.GenPart_genPartIdxMother[iGen]   ## Figure out where the parent is stored
            #    if parentIdx == -1: continue                    ## Skip particles with unknown parents
            #    if ch.GenPart_pdgId[parentIdx] == 25:           ## If a particle has a Higgs parent...
            #        partId = abs(ch.GenPart_pdgId[iGen])
            #        if partId in partList: continue             ## And hasn't appeared yet...
            #        partList.append(partId)                     ## Add it to the pile
            #        print(partList)

            ## Tag muons from A, W, and Z decays for later plotting
            for iGen in range(nGen):
                ## Debug section for analyzing generated muons
                if VERBOSE and ((abs(ch.GenPart_pdgId[iGen]) == 13) or (abs(ch.GenPart_pdgId[iGen]) == 11)):
                    print '>GenPart_pdgId[{}] = {}'.format(iGen,ch.GenPart_pdgId[iGen])
                    print '>GenPart_pt[{}] = {}'.format(iGen,ch.GenPart_pt[iGen])
                    print '>GenPart_eta[{}] = {}'.format(iGen,ch.GenPart_eta[iGen])
                    print '>GenPart_phi[{}] = {}'.format(iGen,ch.GenPart_phi[iGen])
                    parentIdx = ch.GenPart_genPartIdxMother[iGen]
                    if parentIdx != -1: print '>Parent ID = {}'.format(ch.GenPart_pdgId[parentIdx]
)
                if (abs(ch.GenPart_pdgId[iGen]) == 13):             ## Particle is a muon
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
            L1TETA = trigger(25, -1, 999)
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

                if ch.Muon_mediumId[iMu]:                  ## Add in a Medium ID cut to our HLT and L1T requirements
                    if mu_pt >= 24:                     ## If our pt is too large for our trigger object,
                        HLTSIP.store(24,mu_sip)         ## Just fill the top bin
                        L1TETA.store(24,abs(mu_eta))
                    else:
                        HLTSIP.store(int(mu_pt),mu_sip) ## Round each pT down and fill the SIP in that bin
                        L1TETA.store(int(mu_pt),abs(mu_eta))


                goodMu.append(iMu)          ## This muon passes all of our cuts, so we save it


                if VERBOSE and ch.Muon_mediumId[iMu]: 
                    print '  * Muon_charge[{}] = {}'.format(iMu, ch.Muon_charge[iMu])
                    print '  * Muon_pt[{}] = {} GeV'.format(iMu, mu_pt)
                    print '  * Muon_eta[{}] = {}'.format(iMu, mu_eta)
                    print '  * Muon_phi[{}] = {}'.format(iMu, mu_phi)
                    print '  * Muon_sip[{}] = {}'.format(iMu, mu_sip)

                ## Fill the histograms
                plots["h_mu_pt"].cfill(mu_pt)
    
                ## Fill only muons with |eta| < 1.5
                if abs(ch.Muon_eta[iMu]) < 1.5:
                    plots["h_mu_pt_eta"].cfill(mu_pt)                
            ## End loop over RECO muons pairs (iMu)

            ## Begin loop over jets
            goodjets = {}   ## To get the highest key, use max(d); to get the highest value, use max(d, key=d.get)
            for iJet in range(len(ch.Jet_pt)):
                jet_eta = abs(ch.Jet_eta[iJet])
                jet_phi = ch.Jet_phi[iJet]
                jet_pt = ch.Jet_pt[iJet]
                jet_puid = ch.Jet_puId[iJet]
                badJet = False
                ## Only accept jets with |eta| < 2.4
                if (jet_puid is not 0) and (jet_eta < 2.4):
                    ## Check all our W, Z and A candidates to see if our jet is within dR < 0.2 of them
                    for i in range(len(muEta)):
                        if (abs(jet_eta - muEta[i]) < 0.2
                            ) and ((abs(jet_phi - muPhi[i]) < 0.2) or (abs(jet_phi - muPhi[i]) > 6.22)):
                            badJet = True        ## This muon comes from something we're not interested in
                            break
                    ## If so, skip this jet and move on
                    if badJet:
                        continue
                    ## Check that we're not assigning two jets to the same dictionary entry
                    if jet_pt in goodjets:
                        jet_pt = jet_pt + .0000001
                        if jet_pt in goodjets:
                            print("The seriously statistically improbable has occured - Jet pT collision detected")
                            print jet_pt
                    ## Fill the dictionary with the jet
                    goodjets[jet_pt] = jet_eta

            ## Fill cut flow plots based on what was passed (order matters)

            ## Begin trigger and cut analysis
            if promptCut:                           ## This signifies that a muon of interest was even in this event
                plots["h_cutflow_mc"].cfill(1) 
                for plot in trigplots:
                    trigplots[plot].cfill(1)
                for plot in indexplots:
                    indexplots[plot].cfill(1)
                for i in ['Njet_pt15','Njet_pt20','Njet_pt25','Njet_pt30']:
                    plots[i].cfill(-1)
            else: continue                          ## Further trigger work past here only operates on valid events

            stillMu = True
            cutplace = 2
            L1Tpt = [7, 8, 9, 10, 12, 14, 16, 18]           ## These are all the pt cuts for the L1 Trigger
            for i in L1Tpt:                             
                if (L1TETA.check(i) < 1.5) and stillMu:
                    trigplots["Mu"+str(i)+"_ER1p5"].cfill(2)## Fill the appropriate plot with an event
                    trigplots["L1T_cutflow"].cfill(cutplace)## And fill the appropriate cutflow bin
                    cutplace = cutplace + 1
                else:
                    stillMu = False                     ## Once we run out of muons, stop looking

            cutplace = 2
            HLTpt = [7,8,8,8,9,9,9,12]                      ## Combined, these arrays cover our HLT values
            HLTip = [4,3,5,6,4,5,6,6]

            for i in range(len(HLTpt)):                     ## Loop over every HLT
                if (HLTSIP.check(HLTpt[i]) >= HLTip[i]) and (L1TETA.check(HLTpt[i]) < 1.5):
                    trigplots["Mu"+str(HLTpt[i])+"_IP"+str(HLTip[i])].cfill(2)
                    trigplots["HLT_cutflow"].cfill(cutplace)
                    if VERBOSE: print('-------Mu{}_IP{} Status: {}-------'.format(HLTpt[i],HLTip[i], True))
                elif VERBOSE: print('-------Mu{}_IP{} Status: {}-------'.format(HLTpt[i],HLTip[i], False))
                cutplace = cutplace + 1

            ##Replicate individual prescale indeces

            if L1TETA.check(12) < 1.5:
                indexplots["Index3"].cfill(2)
                if HLTSIP.check(12) > 6:
                    indexplots["Index3"].cfill(3)

            if L1TETA.check(10) < 1.5:
                indexplots["Index4"].cfill(2)
                if HLTSIP.check(9) > 5:
                    indexplots["Index4"].cfill(3)

            if L1TETA.check(9) < 1.5:
                indexplots["Index5"].cfill(2)
                if HLTSIP.check(8) > 5:
                    indexplots["Index5"].cfill(3)

            if L1TETA.check(8) < 1.5:
                indexplots["Index6"].cfill(2)
                if HLTSIP.check(7) > 4:
                    indexplots["Index6"].cfill(3)

            if L1TETA.check(7) < 1.5:
                indexplots["Index7"].cfill(2)
                if HLTSIP.check(7) > 4:
                    indexplots["Index7"].cfill(3)

            ## Begin Jet analysis
            i = 0
            njets = [0,0,0,0]
            jetcuts = [15,20,25,30]
            if not goodjets:
                for j in range(4):
                    plots['Njet_pt'+str(jetcuts[j])].cfill(0)
            while (goodjets):## and (i in range(4)):
                jpt = max(goodjets)
                jeta = goodjets.pop(jpt)
                if i < 4:
                    plots['Jet_pt'].cfill(jpt)
                    plots['Jet_eta'].cfill(jeta)
                if i < 6:
                    plots["Jet"+str(i+1)+'_pt'].cfill(jpt)
                for j in range(4):
                    if jpt > jetcuts[j]:
                        njets[j] = njets[j] + 1
                if not goodjets:
                    for j in range(4):
                        plots['Njet_pt'+str(jetcuts[j])].cfill(njets[j])
                i = i + 1
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
    for i in ['Njet_pt15','Njet_pt20','Njet_pt25','Njet_pt30']:
        plots[i].markbins()

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
    for i in ['Njet_pt15','Njet_pt20','Njet_pt25','Njet_pt30']:
        hist = plots.pop(i)
        hist.saveplot(canv, png_dir, drawop="htext")
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

