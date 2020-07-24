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
        
    
def getweights(sigfiles='',LHEBGfiles='',datafiles='',fromfile=True,sigevents='',bgevents='',dataevents=''):
    #%%
    #
    bgfracs = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
    nlhe = len(bgfracs)
    print("Current LHE bg weights are",bgfracs)
    if (sigfiles == '' or LHEBGfiles == '' or datafiles == '') and (fromfile or sigevents == '' or bgevents == '' or dataevents == ''):
        raise ValueError('A complete triplicate of input files, or input events, was not passed properly')
    for sfile in sigfiles:
        for dfile in datafiles:
            if fromfile:
                sigevents = uproot.open(sfile).get('Events')
                bgevents = []
                for i in range(nlhe):
                    bgevents.append(uproot.open(LHEBGfiles[i]).get('Events'))
                dataevents = uproot.open(dfile).get('Events')
            
            sigPU = Hist(200,(-0.5,199.5))
            dataPU= Hist(200,(-0.5,199.5))
                
            sigPU.dfill( pd.DataFrame(sigevents.array( 'PV_npvs')))
            dataPU.dfill(pd.DataFrame(dataevents.array('PV_npvs')))
            
            bgPU = []
            for i in range(nlhe):
                bgPU.append(Hist(200,(-0.5,199.5)))
                bgPU[i].dfill(pd.DataFrame(bgevents[i].array('PV_npvs')))
                
            sigweights = {
                    'genweights': pd.DataFrame(sigevents.array('Generator_weight')).rename(columns=inc),
                    # x-sec * lumi / nEvents
                    'floatNorm': 43.92 * 60000 / sigevents.array('Jet_eta').shape[0],
                    'PUhist': dataPU.divideby(sigPU,split=True),
                    'events': pd.DataFrame(sigevents.array('event')).rename(columns=inc),
                    }
            sigweights.update({'PUweights': pd.DataFrame(np.array(sigweights['PUhist'][0])[sigevents.array('PV_npvs')]).rename(columns=inc)})
            sigweights.update({'normweights': sigweights['floatNorm']*(sigweights['genweights']/sigweights['genweights'])})
            bgweights = []
            for i in range(nlhe):
                bgweights.append({
                        'genweights': pd.DataFrame(bgevents[i].array('Generator_weight')).rename(columns=inc),
                        # data events / (bg events * bg lhe weight)
                        'floatNorm': dataevents.array('Jet_eta').shape[0]/(bgevents[i].array('Jet_eta').shape[0]*bgfracs[i]),
                        'PUhist': dataPU.divideby(bgPU[i],split=True),   
                        'events': pd.DataFrame(bgevents[i].array('event')).rename(columns=inc)
                        })
                bgweights[i].update({'PUweights': pd.DataFrame(np.array(bgweights[i]['PUhist'][0])[bgevents[i].array( 'PV_npvs')]).rename(columns=inc)})
                bgweights[i].update({'normweights': bgweights[i]['floatNorm']*(bgweights[i]['genweights']/bgweights[i]['genweights'])})
                bgname = 'weights/'+fstrip(LHEBGfiles[i])+"-"+fstrip(dfile)+".p"
                print('Writing to',bgname)
                pickle.dump(bgweights,open(bgname, "wb"))
            sgname = 'weights/'+fstrip(sfile)+"-"+fstrip(dfile)+".p"
            print('Writing to',sgname)
            pickle.dump(sigweights,open(sgname, "wb"))
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
