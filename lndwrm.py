#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility lndwrm.py                               ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

#import ROOT as R
#R.gROOT.SetBatch(True)  ## Don't display histograms or canvases when drawn

#import os
#import subprocess
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import itertools as it
#import copy as cp
from analib import Hist, PhysObj, Event, Hist2d, inc
from uproot_methods import TLorentzVector, TLorentzVectorArray
#%%

def ana(files,returnplots=False):
    #%%################
    # Plots and Setup #
    ###################
    
    ## Make a dictionary of histogram objects
    plots = {
        "RjetpT":   Hist(100,(0,100)    ,'RECO matched jet pT','Events','recplots/RjetpT'),
        "Rjeteta":  Hist(66 ,(-3.3,3.3) ,'RECO matched jet eta','Events','recplots/Rjeteta'),
        "RjetCSVV2":Hist([0,0.1241,0.4184,0.7527,1],None,'RECO matched jet btagCSVV2 score','events','recplots/RjetCSVV2'),
        #"RjetDeepB":Hist([0,0.0494,0.2770,0.7264,1],None,'RECO matched jet btagDeepB score','events','recplots/RjetDeepB'),
        "RjetDeepFB":Hist([0,0.0494,0.2770,0.7264,1],None,'RECO matched jet btagDeepFlavB score','events','recplots/RjetDeepFB'),
        "RA1pT":    Hist(80 ,(0,160)    ,'pT of RECO A1 objects constructed from matched jets','Events','recplots/RA1pT'),
        "RA2pT":    Hist(80 ,(0,160)    ,'pT of RECO A2 objects constructed from matched jets','Events','recplots/RA2pT'),
        "RA1mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A1 objects from matched jets','Events','recplots/RA1mass'),
        "RA2mass":  Hist(40 ,(0,80)     ,'reconstructed mass of A2 objects from matched jets','Events','recplots/RA2mass'),
        "RA1dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A1 object','Events','recplots/RA1dR'),
        "RA2dR":    Hist(50 ,(0,5)      ,'dR between jet children of reconstructed A2 object','Events','recplots/RA2dR'),
        "RA1deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A1 object','Events','recplots/RA1deta'),
        "RA2deta":  Hist(33 ,(0,3.3)    ,'|deta| between jet children of reconstructed A2 object','Events','recplots/RA2deta'),
        "RA1dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A1 object','Events','recplots/RA1dphi'),
        "RA2dphi":  Hist(33 ,(0,3.3)    ,'|dphi| between jet children of reconstructed A2 object','Events','recplots/RA2dphi'),
        "RHmass":   Hist(80 ,(0,160)     ,'reconstructed mass of Higgs object from reconstructed As','Events','recplots/RHmass'),
        "RHpT":     Hist(100,(0,200)    ,'pT of reconstructed higgs object from reconstructed As','Events','recplots/RHpT'),
        "RHdR":     Hist(50 ,(0,5)      ,'dR between A children of reconstructed higgs object','Events','recplots/RHdR'),
        "RHdeta":   Hist(33 ,(0,3.3)    ,'|deta| between A children of reconstructed higgs object','Events','recplots/RHdeta'),
        "RHdphi":   Hist(33 ,(0,3.3)    ,'|dphi| between A children of reconstructed higgs object','Events','recplots/RHdphi'),
        ##
        "RalljetpT":    Hist(100,(0,100),'All RECO jet pT','Events','recplots/RalljetpT'),
        "npassed":  Hist(1  ,(0.5,1.5) ,'','Number of events that passed cuts', 'recplots/npassed')
    }
    for plot in plots:
        plots[plot].title = files[0]

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    
    ## Loop over input files
    for fnum in range(len(files)):
        
        #####################
        # Loading Variables #
        #####################
        
        print('Opening '+files[fnum])
        ## Open our file and grab the events tree
        f = uproot.open(files[fnum])#'nobias.root')
        events = f.get('Events')

        jets = PhysObj('jets')

        jets.eta= pd.DataFrame(events.array('Jet_eta')).rename(columns=inc)
        jets.phi= pd.DataFrame(events.array('Jet_phi')).rename(columns=inc)
        jets.pt = pd.DataFrame(events.array('Jet_pt')).rename(columns=inc)
        jets.mass=pd.DataFrame(events.array('Jet_mass')).rename(columns=inc)
        jets.CSVV2 = pd.DataFrame(events.array('Jet_btagCSVV2')).rename(columns=inc)
        #jets.DeepB = pd.DataFrame(events.array('Jet_btagDeepB')).rename(columns=inc)
        jets.DeepFB= pd.DataFrame(events.array('Jet_btagDeepFlavB')).rename(columns=inc)

        print('Processing ' + str(len(jets.eta)) + ' events')

        ## Figure out how many bs and jets there are
        njet= jets.eta.shape[1]

        ev = Event(jets)
        jets.cut(jets.pt>15)
        jets.cut(abs(jets.eta)<2.4)
        ev.sync()
        
        
        ##############################
        # Processing and Calculation #
        ##############################

        ## Create our dR dataframe by populating its first column and naming it accordingly
        jjdr2 = pd.DataFrame(np.power(jets.eta[1]-jets.eta[2],2) + np.power(jets.phi[1]-jets.phi[2],2)).rename(columns={0:'Jet 1 x Jet 2'})
        jjmass = pd.DataFrame(jets.mass[1] + jets.mass[2]).rename(columns={0:'Jet 1 x Jet 2'})
        
        ## Loop over jet x b combinations
        jjstr = []
        for j in range(1,njet+1):
            for i in range(j+1,njet+1):
                ## Make our column name
                jjstr.append("Jet "+str(j)+" x Jet "+str(i))
                #if (i+j == 3):
                #    continue
                ## Compute and store the dr of the given b and jet for every event at once
                jjdr2[jjstr[-1]] = pd.DataFrame(np.power(jets.eta[j]-jets.eta[i],2) + np.power(jets.phi[j]-jets.phi[i],2))
                jjmass[jjstr[-1]] = pd.DataFrame(jets.mass[j] + jets.mass[i])
                #if (j==i):
                #    jjdr2[jjstr[-1]] = jjdr2[jjstr[-1]] * np.nan
                #    jjmass[jjstr[-1]] = jjmass[jjstr[-1]] * np.nan
                    
        print('jjs done')
                    
        j4mass = pd.DataFrame(jets.mass[1]+jets.mass[2]+jets.mass[3]+jets.mass[4]).rename(columns={0:"J1 x J2 J3 J4"})
        j4str = []
        for a in range(1,njet+1):
            for b in range(a+1,njet+1):
                for c in range(b+1,njet+1):
                    for d in range(c+1,njet+1):
                        j4str.append("J"+str(a)+" J"+str(b)+" J"+str(c)+" J"+str(d))
                        #if (a+b+c+d == 10):
                        #    continue
                        j4mass[j4str[-1]] = pd.DataFrame(jets.mass[a]+jets.mass[b]+jets.mass[c]+jets.mass[d])
                        #if (a==b or a==c or a==d or b==c or b==d or c==d):
                        #    j4mass[j4str[-1]] = j4mass[j4str[-1]] * np.nan
                       
        print('j4s done')
        
        ## Create a copy array to collapse in jets into
        drlist = []
        mmlist = []
        m4list = []
        for j in range(njet):
            drlist.append(np.sqrt(jjdr2.filter(like='Jet '+str(j+1))))
            mmlist.append(jjmass.filter(like='Jet '+str(j+1)))
            m4list.append(j4mass.filter(like='J'+str(j+1)))
            #jlist[j] = jlist[j][jlist[j].rank(axis=1,method='first') == 1]
            #drlist[j] = jlist[j].rename(columns=lambda x:int(x[4:6]))
            #mmlist[j] = mmlist[j].rename(columns=lambda x:int(x[4:6]))
            
        print('jlist done')

        ## Cut our events to only resolved 4jet events with dR<0.4
        djets = mmlist[0][(mmlist[0]>25) & (mmlist[0]<65)]
        qjets = m4list[0][(m4list[0]>90) & (m4list[0]<150)]
        for i in range(1,njet):
            djets = djets.combine_first(mmlist[i][(mmlist[i]>25) & (mmlist[i]<65)])
            qjets = qjets.combine_first(m4list[i][(m4list[i]>90) & (m4list[i]<150)])
        djets = djets / djets
        qjets = qjets / qjets
        djets = djets.sum(axis=1)
        qjets = qjets.sum(axis=1)
        djets = djets[djets>=2].dropna()
        qjets = qjets[qjets>=1].dropna()
        jets.trimTo(djets)
        jets.trimTo(qjets)
        ev.sync()
        
        print('trimming done')
        
        #############################
        # Constructing RECO objects #
        #############################


        for prop in ['bpt','beta','bphi','bmass']:
            jets[prop] = pd.DataFrame()
            for i in range(1,5):
                jets[prop][i] = jets[prop[1:]][jets.mass.rank(axis=1,method='first',ascending=False) == i].max(axis=1)
                
        #jets.bdr = pd.DataFrame()
        #for i in range(nb):
        #    jets.bdr[i+1] = blist[i][blist[i]>0].max(axis=1)
            
        #ev.sync()
            
                    
        bvec = []
        for i in range(1,5):
            bvec.append(TLorentzVectorArray.from_ptetaphim(jets.bpt[i],jets.beta[i],jets.bphi[i],jets.bmass[i]))
        
        avec = []
        for i in range(0,4,2):
            avec.append(bvec[i]+bvec[i+1])
        
        for prop in ['apt','aeta','aphi','amass']:
            jets[prop] = pd.DataFrame()
        for i in range(2):
            jets.apt[i+1]  = avec[i].pt
            jets.aeta[i+1] = avec[i].eta
            jets.aphi[i+1] = avec[i].phi
            jets.amass[i+1]= avec[i].mass
        for prop in ['apt','aeta','aphi','amass']:
            jets[prop].index = jets.pt.index
        
        hvec = [avec[0]+avec[1]]
        
        for prop in ['hpt','heta','hphi','hmass']:
            jets[prop] = pd.DataFrame()
        jets.hpt[1]  = hvec[0].pt
        jets.heta[1] = hvec[0].eta
        jets.hphi[1] = hvec[0].phi
        jets.hmass[1]= hvec[0].mass
        for prop in ['hpt','heta','hphi','hmass']:
            jets[prop].index = jets.eta.index
        
        ################
        # Filling Data #
        ################
        
        plots['RalljetpT'].dfill(jets.pt)
        #plots['bjdR'].dfill(jets.bdr)
        plots['RjetpT'].dfill(jets.bpt)
        plots['Rjeteta'].dfill(jets.beta)  
        for i in range(1,3):
            plots['RA'+str(i)+'pT'  ].fill(jets.apt[i])
            plots['RA'+str(i)+'mass'].fill(jets.amass[i])
            #lots['RA'+str(i)+'deta'].fill(abs(jets.beta[2*i]-jets.beta[(2*i)-1]))
            plots['RA'+str(i)+'dR'  ].fill(np.sqrt(np.power(jets.beta[2*i]-jets.beta[(2*i)-1],2)+np.power(jets.bphi[2*i]-jets.bphi[(2*i)-1],2)))
            plots['RA'+str(i)+'deta'].fill(abs(jets.beta[2*i]-jets.beta[(2*i)-1]))
            plots['RA'+str(i)+'dphi'].fill(abs(jets.bphi[2*i]-jets.bphi[(2*i)-1]))
        plots['RHpT'  ].fill(jets.hpt[1])
        plots['RHmass'].fill(jets.hmass[1])
        plots['RHdR'  ].fill(np.sqrt(np.power(jets.aeta[2]-jets.aeta[1],2)+np.power(jets.aphi[2]-jets.aphi[1],2)))
        plots['RHdeta'].fill(abs(jets.aeta[2]-jets.aeta[1]))
        plots['RHdphi'].fill(abs(jets.aphi[2]-jets.aphi[1]))
        plots['npassed'].fill(jets.hpt[1]/jets.hpt[1])
        plots['RjetCSVV2'].dfill(jets.CSVV2)
        plots['RjetDeepFB'].dfill(jets.DeepFB)
    
    ############
    # Plotting #
    ############
        
    plt.clf()
    #plots.pop('bjdR').plot(logv=True)

    for p in plots:
        plots[p].plot()
    #%%
    if returnplots==True:
        return plots
    else:
        sys.exit()

def trig(files):
    ## Create a dictionary of histogram objects
    rptbins = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,30,100]
    plots = {
        'hptplot':      Hist(rptbins,None,'Highest Muon pT','Events passing HLT','recplots/TrigHpTplot'),
        'ptplot':       Hist(rptbins,None,'Highest Muon pT','Events','recplots/TrigpTplot'),
        'ratioptplot':  Hist(rptbins,None,'Highest Muon pT','HLT_Mu7_IP4 / Events with Muons of sip > 5','recplots/TrigRatiopTPlot'),
        'sipplot':      Hist(20,(0,20),'Highest Muon SIP', 'Events', 'recplots/TrigSIPplot'),
        'hsipplot':     Hist(20,(0,20),'Highest Muon SIP', 'Events', 'recplots/TrigHSIPplot'),
        'ratiosipplot': Hist(20,(0,20),'Highest Muon SIP', 'HLT_Mu7_IP4 / Events with muons of pT > 10', 'recplots/TrigRatioSIPplot'),
        'HLTcutflow':      Hist(12,(-0.5,11.5),'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6','Events','recplots/cutflowHLT'),
        'L1Tcutflow':      Hist(12,(-0.5,11.5),'All // L1_SingleMu6/7/8/9/10/12/14/16/18','Events','recplots/cutflowL1T'),
        'HLTcutflowL':      Hist(12,(-0.5,11.5),'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6','Events','recplots/cutflowHLT-L'),
        'L1TcutflowL':      Hist(12,(-0.5,11.5),'All // L1_SingleMu6/7/8/9/10/12/14/16/18','Events','recplots/cutflowL1T-L')

    }
    cutflow2d = Hist2d([9,10],[[-0.5,8.5],[-0.5,9.5]],'All // HLT_Mu7/8/9/12_IP4/3,5,6/4,5,6/6',
        'All // L1_SingleMu6/7/8/9/10/12/14/16/18','recplots/cutflowHLTvsL1T',files[0])
    for plot in plots:
        plots[plot].title = files[0]
    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    ## Loop over all input files
    for fnum in range(len(files)):
        print('Opening '+files[fnum])
        ## Open the file and retrieve our key branches
        f = uproot.open(files[fnum])
        events = f.get('Events')

        HLTcuts = ['HLT_Mu7_IP4','HLT_Mu8_IP3','HLT_Mu8_IP5','HLT_Mu8_IP6','HLT_Mu9_IP4','HLT_Mu9_IP5','HLT_Mu9_IP6','HLT_Mu12_IP6']
        L1Tcuts = ['L1_SingleMu6','L1_SingleMu7','L1_SingleMu8','L1_SingleMu9','L1_SingleMu10','L1_SingleMu12','L1_SingleMu14','L1_SingleMu16','L1_SingleMu18']

        Muon = PhysObj('Muon',files[fnum],'pt','eta','phi','sip3d','mediumId')
        Trig = PhysObj('trig')
        HLT = PhysObj('HLTrig')
        L1T = PhysObj('L1Trig')
        Trig.vals = pd.DataFrame(events.array('HLT_Mu7_IP4_part0')).rename(columns=inc)
        for tr in HLTcuts:
            HLT[tr] = pd.DataFrame(events.array(tr+'_part0')).rename(columns=inc)
        for tr in L1Tcuts:
            L1T[tr]= pd.DataFrame(events.array(tr+'er1p5')).rename(columns=inc)
        ev = Event(Muon,Trig,HLT,L1T)
        print('Processing ' + str(len(Muon.pt)) + ' events')
   
        ## Fill 0 bin of cut flow plots

        plots['HLTcutflow'].dfill(HLT[HLTcuts[0]]*0)
        plots['L1Tcutflow'].dfill(L1T[L1Tcuts[0]]*0)
        cutflow2d.dfill(HLT[HLTcuts[0]]*0,HLT[HLTcuts[0]]*0)

 
        ## Fill the rest of the bins
        ct = 1
        for i in HLT:
            plots['HLTcutflow'].dfill(HLT[i][HLT[i]].dropna()*ct)
            cutflow2d.dfill(HLT[i][HLT[i]].dropna()*ct,HLT[i][HLT[i]].dropna()*0)
            ct = ct + 1
        ct = 1
        for i in L1T:
            plots['L1Tcutflow'].dfill(L1T[i][L1T[i]].dropna()*ct)
            cutflow2d.dfill(L1T[i][L1T[i]].dropna()*0,L1T[i][L1T[i]].dropna()*ct)
            ct = ct + 1

        ht = 1
        for i in HLT:
            lt = 1
            for j in L1T:
                cutflow2d.dfill(HLT[i][HLT[i] & L1T[j]].dropna()*ht,L1T[j][L1T[j] & HLT[i]].dropna()*lt)
                lt = lt + 1
            ht = ht + 1

        ##Perform global cuts
        Muon.cut(abs(Muon.eta)<1.5)
        Muon.cut(Muon.mediumId==True)
        ev.sync()

        ##Fill bin 1 of cut flow lots

        #plots['HLTcutflow'].fill((Muon.pt/Muon.pt).max(axis=1).dropna())
        #plots['L1Tcutflow'].fill((Muon.pt/Muon.pt).max(axis=1).dropna())


        ## Cut muons and trim triggers to the new size
        MuonP = Muon.cut(Muon.sip3d>5,split=True)
        MuonS = Muon.cut(Muon.pt>10,split=True)
        TrigP = Trig.trimTo(MuonP.pt,split=True)
        TrigS = Trig.trimTo(MuonS.sip3d,split=True)
        ## Reshape triggers to fit our muons
        for i in MuonP.pt.columns:
            TrigP.vals[i] = TrigP.vals[1]
        for i in MuonS.sip3d.columns:
            TrigS.vals[i] = TrigS.vals[1]

        ## Create the two histograms we want to divide
        plt.figure(1)
        plots['ptplot'].fill(MuonP.pt.max(axis=1))
        plots['hptplot'].fill(MuonP.pt[TrigP.vals].max(axis=1).dropna(how='all'))
        plots['sipplot'].fill(MuonS.sip3d.max(axis=1))
        plots['hsipplot'].fill(MuonS.sip3d[TrigS.vals].max(axis=1).dropna(how='all'))
    plots['ratioptplot'].add(plots['hptplot'].divideby(plots['ptplot'],split=True))
    plots['ratiosipplot'].add(plots['hsipplot'].divideby(plots['sipplot'],split=True))
    plots['HLTcutflowL'].add(plots['HLTcutflow'])
    plots['L1TcutflowL'].add(plots['L1Tcutflow'])
    cutflow2d.norm()[0][0][0] = 0
    cutflow2d.plot(text=True,edgecolor='black')
    plots.pop('HLTcutflowL').norm().plot(ylim=(None,.2))
    plots.pop('L1TcutflowL').norm().plot(ylim=(None,.2))
    plots.pop('HLTcutflow').norm().plot()
    plots.pop('L1Tcutflow').norm().plot()

    for pl in plots:
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
        elif sys.argv[1] == '-trig':
            trig(files)
        elif sys.argv[1] == '-a':
            ana(files)
 
    print("Expected n00dle.py <switch> (flag) (target)")
    print("-----switches-----")
    print("-trig  Analyzes trigger efficiency for data")
    print("-a     Analyzes jet-b correlations")
    print("---optional flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)
    

## Define 'main' function as primary executable
if __name__ == '__main__':
    main()
