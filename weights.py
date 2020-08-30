#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
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
from analib import Hist, PhysObj, Event, inc, fstrip#, Hist2D
import pickle
import copy as cp


import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()

##The weights LHE segment split data should be merged by
#bgweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
#nlhe = len(bgweights)

##########################################################
# normevts = x-sec * lumi for signal, or # events for bg #
        
## Previously used for finding floating point division errors
def debugsearch(hist,vhist,dhist):
    for idx in range(len(hist[0])):
        if hist[0][idx] > 5:
            print(vhist.title)
            print('ratio:',hist[0][idx])
            print('value:',vhist[0][idx])
            print('data:',dhist[0][idx])

def getweights(sigfiles='',LHEBGfiles='',datafiles='',fromfile=True,sigevents='',bgevents='',dataevents=''):
    #%%######################
    ## Opening Definitions ##
    #########################
    bgfracs = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
    nlhe = len(bgfracs)
    print("Current LHE bg weights are",bgfracs)
    if (sigfiles == '' or LHEBGfiles == '' or datafiles == '') and (fromfile or sigevents == '' or bgevents == '' or dataevents == ''):
        raise ValueError('A complete triplicate of input files, or input events, was not passed properly')
    for sfile in sigfiles:
        for dfile in datafiles:
            ## Handles file operations if event trees are not passed
            if fromfile:
                sigevents = uproot.open(sfile).get('Events')
                bgevents = []
                for i in range(nlhe):
                    bgevents.append(uproot.open(LHEBGfiles[i]).get('Events'))
                dataevents = uproot.open(dfile).get('Events')
            
            ## Initializing, filling, and normalizing histograms
            sigPU = Hist(200,(-0.5,199.5),'event pileup','fraction of events','weightplots/sigPU','Signal Pileup')
            dataPU= Hist(200,(-0.5,199.5),'event pileup','fraction of events','weightplots/dataPU','Data Pileup')
            compPU= Hist(200,(-0.5,199.5),'normalized signal (red), data (black), and weighted signal (blue)','bin fraction','weightplots/compPU')
                
            sigPU.dfill( pd.DataFrame(sigevents.array( 'PV_npvs')))
            dataPU.dfill(pd.DataFrame(dataevents.array('PV_npvs')))
            
            
            sigPU[0] = sigPU[0] / sum(sigPU[0])
            dataPU[0] = dataPU[0] / sum(dataPU[0])
            
            bgPU = []
            bgesum = 0
            ## In addition to looping previous operations over LHE background slices,
            ## The total background event number after fraction cuts is calculated
            for i in range(nlhe):
                bgPU.append(Hist(200,(-0.5,199.5),'event pileup','fraction of events','weightplots/bgPU_'+str(i),'Background #'+str(i)+' Pileup'))
                bgPU[i].dfill(pd.DataFrame(bgevents[i].array('PV_npvs')))
                bgPU[i][0] = bgPU[i][0] / sum(bgPU[i][0])
                
                ## Pre-selection cuts must be applied to find surviving QCD event number
                #bgjets = PhysObj('FatJet',LHEBGfiles[i],'pt','eta','mass','msoftdrop','btagDDBvL','btagDeepB')
                #bgjets.cut(bgjets.pt > 240).cut(abs(bgjets.eta)<2.4).cut(bgjets.btagDDBvL > 0.8).cut(
                        #bgjets.btagDeepB > 0.4184).cut(bgjets.msoftdrop > 90).cut(bgjets.mass > 90)
                ## Tallying event number after cuts
                #bgesum = bgesum + (bgjets.pt.shape[0] * bgfracs[i])

            ## Applying pre-selection cuts to data to find total passing event number
            #dtjets = PhysObj('FatJet',dfile,'pt','eta','mass','msoftdrop','btagDDBvL','btagDeepB')
            #dtjets.cut(dtjets.pt > 240).cut(abs(dtjets.eta)<2.4).cut(dtjets.btagDDBvL > 0.8).cut(
                    #dtjets.btagDeepB > 0.4184).cut(dtjets.msoftdrop > 90).cut(dtjets.mass > 90)
            
            ####################
            ## Signal Weights ##
            ####################
            
            sigweights = {
                    'genweights': pd.DataFrame(sigevents.array('Generator_weight')).rename(columns=inc),
                    # x-sec * lumi / nEvents
                    'floatNorm': 43.92 * 60000 / sigevents.array('Jet_eta').shape[0],
                    'PUhist': dataPU.divideby(sigPU,split=True,trimnoise=.00001),
                    'events': pd.DataFrame(sigevents.array('event')).rename(columns=inc),
                    }
            sigweights.update({'PUweights': pd.DataFrame(np.array(sigweights['PUhist'][0])[sigevents.array('PV_npvs')]).rename(columns=inc)})
            sigweights.update({'normweights': sigweights['floatNorm']*(sigweights['genweights']/sigweights['genweights'])})

            ## Test histograms for inspection purposes
            tPU = sigweights['PUhist']
            tPU.fname = "weightplots/sigDdataPU"
            tPU.title = 'Data / Signal Pileup'
            tPU.ylabel = 'per-bin event weighting'
            tPU.plot()
            sigPU.plot()
            dataPU.plot()
            
            compPU.fill(sigevents.array( 'PV_npvs'),weights=sigweights['PUweights'][1])
            compPU[0] = compPU[0]/sum(compPU[0])
            plt.clf()
            sigPU.make(color='red'  ,linestyle='-',htype='step')
            dataPU.make(color='black',linestyle='--',htype='step')
            compPU.plot(same=True,color='blue' ,linestyle=':',htype='step')
            
            ratioPUs=[Hist(200,(-0.5,199.5),'event pileup before (red) and after (black) weighting','','weightplots/ratioPU')]
            ratioPUs.append(cp.deepcopy(ratioPUs[0]))
            ratioPUs[0].fill(sigevents.array( 'PV_npvs'))
            rawint = ratioPUs[0][0].sum()
            ratioPUs[1].fill(sigevents.array( 'PV_npvs'),weights=sigweights['PUweights'][1])
            wgtint = round(ratioPUs[1][0].sum())
            ratioPUs[1].title='unweighted area '+str(rawint)+', weighted area '+str(wgtint)
            plt.clf()
            ratioPUs[0].make(color='red'  ,linestyle='-',htype='step')
            ratioPUs[1].plot(same=True,color='black',linestyle='--',htype='step')
            
            ########################
            ## Background Weights ##
            ########################
            
            bgweights = []
            for i in range(nlhe):
                bgweights.append({
                        'genweights': pd.DataFrame(bgevents[i].array('Generator_weight')).rename(columns=inc),
                        # data events / (bg events * bg lhe weight)
                        'floatNorm': 1,#dtjets.pt.shape[0]/bgesum,#*bgfracs[i]),
                        'PUhist': dataPU.divideby(bgPU[i],split=True,trimnoise=.00001),   
                        'events': pd.DataFrame(bgevents[i].array('event')).rename(columns=inc)
                        })
                bgweights[i].update({'PUweights': pd.DataFrame(np.array(bgweights[i]['PUhist'][0])[bgevents[i].array( 'PV_npvs')]).rename(columns=inc)})
                bgweights[i].update({'normweights': bgweights[i]['floatNorm']*(bgweights[i]['genweights']/bgweights[i]['genweights'])})
                bgname = 'weights/'+fstrip(LHEBGfiles[i])+"-"+fstrip(dfile)

                ## Saving BG output
                print('Writing to',bgname+".p")
                np.savetxt(bgname+'.txt',bgweights[i]['PUhist'][0])
                pickle.dump(bgweights[i],open(bgname+'.p', "wb"))
                
                ## Test histograms for inspection purposes
                tPU = bgweights[i]['PUhist']
                tPU.fname = "weightplots/bgDdataPU_"+str(i)
                tPU.title = 'Data / Background slice '+str(i)+' Pileup'
                tPU.ylabel = 'per-bin event weighting'
                tPU.plot()
                bgPU[i].plot()

                
            ## Saving signal output
            sgname = 'weights/'+fstrip(sfile)+"-"+fstrip(dfile)
            print('Writing to',sgname+".p")
            pickle.dump(sigweights,open(sgname+'.p', "wb"))
            np.savetxt(sgname+'.txt',sigweights['PUhist'][0])
            
            return sigweights, bgweights
                
## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        nrgs = len(sys.argv)
        sigfiles = []
        bgfiles = []
        datafiles = []
        ## Check for file sources
        for i in range(nrgs):
            arg = sys.argv[i]
            if '-f' in arg:
                if 's' in arg:
                    fileptr = sigfiles
                elif 'b' in arg:
                    fileptr = bgfiles
                elif 'd' in arg:
                    fileptr = datafiles
                else:
                    dialogue()
                for j in range(i+1,nrgs):
                    if '-' in sys.argv[j]:
                        break
                    else:
                        fileptr.append(sys.argv[j])
                        i = j
            elif '-l' in arg:
                if 's' in arg:
                    fileptr = sigfiles
                elif 'b' in arg:
                    fileptr = bgfiles
                elif 'd' in arg:
                    fileptr = datafiles
                else:
                    dialogue()
                for j in range(i+1,nrgs):
                    if '-' in sys.argv[j]:
                        break
                    else:
                        with open(sys.argv[j],'r') as rfile:
                            for line in rfile:
                                fileptr.append(line.strip('\n'))
                        i = j
        if (len(sigfiles) and len(bgfiles) and len(datafiles)):        
            getweights(sigfiles,bgfiles,datafiles)
        else:
            dialogue()

    else:
        dialogue()
        
def dialogue():
    print("Expected: weights.py <-f/-l>s (signal.root) <-f/-l>b (background.root) <-f/-l>d (data.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    print("s      Marks the following file(s) as signal")
    print("b      Marks the following file(s) as background")
    print("d      Marks the following file(s) as data")
    sys.exit(0)
    
if __name__ == "__main__":
    main()

#%%

#X = pd.DataFrame()
#X['x'] = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
#X['y'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#X['z'] = [5,3,4,1,2,4,5,2,3,4,2,4,5,5,5,5,3,4,2,2]
#prop = ['x','y','z']
#Y = pd.DataFrame()
#Y['val']=[1,0,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,0,1]
#y = Y['val']
#
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(8,)),
#    keras.layers.Dense(16, activation=tf.nn.relu),
#    keras.layers.Dense(16, activation=tf.nn.relu),
#    keras.layers.Dense(1, activation=tf.nn.sigmoid),
#])
#
#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#model.fit(X, Y, epochs=150, batch_size=1)
#
#test_loss, test_acc = model.evaluate(X_test, y_test)
#print('Test accuracy:', test_acc)
##%%
#print(model.predict(np.array([[1,1,5],[2,1,4],[3,1,3],[4,1,2],[5,1,1]])))
#print(model.predict(np.array([[1,3,1],[2,5,4],[3,3,3],[4,5,2],[4,3,4]])))
