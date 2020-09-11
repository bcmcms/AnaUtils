#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility mndwrm.py                               ###
### Compiled with Keras-2.4.3 Tensorflow-1.14.0                      ###
###                                                                  ###
### Run without arguments for a list of flags and options            ###
########################################################################

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
#from uproot_methods import TLorentzVector, TLorentzVectorArray
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.keras import backend as BE
from keras import backend as K

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':True,'legend.fontsize':14,'legend.edgecolor':'black','hatch.linewidth':1.0})
#plt.style.use({"font.size": 14})
#plt.style.use(hep.cms.style.ROOT)


import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()

##Controls how many epochs the network will train for
epochs = 50
##The weights LHE segment split data should be merged by
lheweights = [1,0.259,0.0515,0.01666,0.00905,0.003594,0.001401]
nlhe = len(lheweights)
##Controls whether a network is trained up or loaded from disc
LOADMODEL = True
##Switches tutoring mode on or off
TUTOR = False
##Switches whether training statistics are reported or suppressed (for easier to read debugging)
VERBOSE=False
##Switches whether weights are loaded and applied to the post-training statistics,
##and what data file they expect to be associated with
POSTWEIGHT = True
DATANAME = '2018D_Parked.root'

evtlist = [35899001,24910172,106249475,126514437,43203653,27186346,17599588,64962950,61283040,54831588]

def binary_focal_loss(alpha=.25, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
    

def tutor(bgjetframe,sigjetframe):
    bgtrnframe = bgjetframe.sample(frac=0.7,random_state=6)
    #nbg = bgtestframe.shape[0]
    sigtrnframe = sigjetframe.sample(frac=0.7,random_state=6)
    #nsig = sigtestframe.shape[0]

    scaler = MinMaxScaler()
    X_test = pd.concat([bgjetframe.drop(bgtrnframe.index), sigjetframe.drop(sigtrnframe.index)],ignore_index=True)
    Y_test = X_test['val']
    X_test = X_test.drop('val',axis=1)
    X_test = scaler.fit_transform(X_test)
    
    records = {}
    rsums = {}
    lr=.01
    for l1 in [7,8]:
        for l2 in [7,8]:
            for l3 in [7,8]:            
                for alpha in [0.5,0.6,0.7,0.8,0.85,0.9]:
                    for gamma in [0.6,0.7,0.8,0.85,0.9,1.0,1.2]:
                        rname = str(l1)+' '+str(l2)+' '+str(l3)+': alpha '+str(alpha)+' gamma '+str(gamma)
                        aoc = []
                        for i in range(10):
                            #tf.random.set_random_seed(2)
                            #tf.compat.v1.random.set_random_seed(2)
                            #tf.compat.v1.set_random_seed(2)
                            np.random.seed(2)
                            model = keras.Sequential([
                                    keras.Input(shape=(8,),dtype='float32'),
                                    #keras.layers.Flatten(input_shape=(8,)),
                                    keras.layers.Dense(l1, activation=tf.nn.relu),
                                    keras.layers.Dense(l2, activation=tf.nn.relu),
                                    keras.layers.Dense(l3, activation=tf.nn.relu),
                                    keras.layers.Dense(1, activation=tf.nn.sigmoid),
                                    ])
                            optimizer  = keras.optimizers.Adam(learning_rate=lr)
                            model.compile(optimizer=optimizer,     
                                          #loss='binary_crossentropy',
                                          #loss=[focal_loss],
                                          #loss=[custom],
                                          loss=[binary_focal_loss(alpha, gamma)],
                                          metrics=['accuracy'])#,tf.keras.metrics.AUC()])

                            X_train = pd.concat([bgtrnframe,sigtrnframe],ignore_index=True)
                            Y_train= X_train['val']
                            X_train = X_train.drop('val',axis=1)
                            X_train = scaler.transform(X_train)
                            model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True)

                            rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
                            aoc.append(auc(rocx,rocy))
                        aocsum = np.sum(aoc)/10               
                        records[rname] = aocsum
                        rsums[rname] = aoc
                        print(rname,' = ',aocsum)
                    
    winner = max(records,key=records.get)
    print('Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    winner = max(records,key=records.get)
    print('2nd Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    winner = max(records,key=records.get)
    print('3rd Best performance was AOC of ',records.pop(winner),' for layout ',winner)
    print(rsums[winner])
    sys.exit()
    

#%%

def ana(sigfiles,bgfiles,isLHE=False,dataflag=False):
    #%%################
    # Plots and Setup #
    ###################
    #training=True
    #training=False
    #tf.random.set_random_seed(2)
    #tf.compat.v1.set_random_seed(2)
    #np.random.seed(2)
    
    #fig = plt.figure(figsize=(10.0,6.0))
    
    global LOADMODEL
    plargs = {'Data':  {'color':'black','htype':'step'},
              'Background':   {'color':'red','htype':'step'},
              'Signal':{'color':'blue','htype':'step'}
              }
    Skey = 'Signal'
    Bkey = 'Background'
    ## The dataflag controls which file list if any has been replaced by data
    if dataflag:
        LOADMODEL = True
        if dataflag == True:
            plargs['Data'].update({'htype':'err'})
            plargs['Background'].update( {'htype':'bar'})
            Skey = 'Data'
            Bkey = 'Background'
        else:
            Skey = 'Signal'
            Bkey = 'Data'

    
    l1 = 8
    l2 = 8
    l3 = 8
    alpha = 0.85
    gamma = 0.8
    model = keras.Sequential([
            #keras.Input(shape=(4,),dtype='float32'),
            #keras.layers.Flatten(input_shape=(8,)),
            keras.layers.Dense(l1, activation=tf.nn.relu,input_shape=(4,)),#, bias_regularizer=tf.keras.regularizers.l2(l=0.0)),
            keras.layers.Dense(l2, activation=tf.nn.relu),
            keras.layers.Dense(l3, activation=tf.nn.relu),
            #keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ])
    optimizer  = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,#'adam',     
                  #loss='binary_crossentropy',
                  #loss=[focal_loss],
                  #loss=[custom],
                  loss=[binary_focal_loss(alpha, gamma)],
                  metrics=['accuracy'])#,tf.keras.metrics.AUC()])
            
    #nbatch = math.floor(nbg / (2*nsig))
    
    scaler = MinMaxScaler()
    
    #netvars = ['pt','eta','mass','CSVV2','DeepB','msoft','DDBvL','H4qvs']
    netvars = ['DeepB','H4qvs','DDBvL','CSVV2']
    #netvars = ['pt','eta','mass','msoft']
    
    ## Define what pdgId we expect the A to have
    Aid = 9000006
    ## How many resolved jets we want to target with our analysis
    #resjets = 4
    Aid = 36
    ## Make a dictionary of histogram objects
    plots = {
        "Distribution": Hist(50,(0,1),'Confidence','Fraction of Events','netplots/Distribution'),
        "DistributionL": Hist(50,(0,1),'Confidence','Fraction of Events','netplots/LogDistribution'),
        "DistStr":  Hist(50,(0,1)),
        "DistSte":  Hist(50,(0,1)),
        "DistBtr":  Hist(50,(0,1)),
        "DistBte":  Hist(50,(0,1)),
        "LossvEpoch":   Hist(epochs,(0.5,epochs+.5),'Epoch Number','Loss','otherplots/LossvEpoch'),
        "AccvEpoch":Hist(epochs,(0.5,epochs+.5),'Epoch Number','Accuracy','otherplots/AccvEpoch'),
    }
    if POSTWEIGHT:
        plots.update({
                "WeightStr":Hist(40,(0,5)),
                "WeightSte":Hist(40,(0,5)),
                "WeightBtr":Hist(40,(0,5)),
                "WeightBte":Hist(40,(0,5)),
                "Weights": Hist(40,(0,5),'postweight values','# of weights','netplots/Weights')})
    
    pplots = {
        "pt":       Hist(80 ,(150,550)  ,'pT for highest pT jet','Fractional Distribution','netplots/ppt'),
        "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT jet','Fractional Distribution','netplots/peta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/pphi'),
        "mass":     Hist(50 ,(0,200)    ,'mass for highest pT jet','Fractional Distribution','netplots/pmass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet in all (red), passing signal (blue), and signal (black) events','Fractional Distribution','netplots/pCSVV2'),
        "DeepB":    Hist(22 ,(0,1.1)    ,'DeepB for highest pT jet','Fractional Distribution','netplots/pDeepB'),
        "msoft":    Hist(50 ,(0,200)    ,'msoft for highest pT jet in all (red), passing (blue), and failing  (black) events','Fractional Distribution','netplots/pmsoft'),
        "DDBvL":    Hist(55 ,(0,1.1)    ,'DDBvL for highest pT jet','Fractional Distribution','netplots/pDDBvL'),
        "H4qvs":    Hist(24 ,(-10,2)    ,'H4qvs for highest pT jet','Fractional Distribution','netplots/pH4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/pnpvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/pnpvsG'),
        }
    prefix = ['SG','SPS','SFL','BG','BPS','BFL']
    tdict = {}
    for plot in pplots:
        size = pplots[plot].size
        bounds = pplots[plot].bounds
        for fix in prefix:
            tdict.update({fix+plot:Hist(size,bounds)})
        tdict.update({'B'+plot:cp.deepcopy(pplots[plot])})
        tdict['B'+plot].fname = tdict['B'+plot].fname+'B'
    pplots.update(tdict)
    del tdict
    
    vplots = {
        "pt":       Hist(80 ,(150,550)  ,'pT for highest pT jet','Fractional Distribution','netplots/pt'),
        "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT jet','Fractional Distribution','netplots/eta'),
        "phi":      Hist(32 ,(-3.2,3.2) ,'phi for highest pT jet','Fractional Distribution','netplots/phi'),
        "mass":     Hist(50 ,(0,200)    ,'mass for highest pT jet','Fractional Distribution','netplots/mass'),
        "CSVV2":    Hist(22 ,(0,1.1)    ,'CSVV2 for highest pT jet','Fractional Distribution','netplots/CSVV2'),
        "DeepB":    Hist(22 ,(0,1.1)    ,'DeepB for highest pT jet','Fractional Distribution','netplots/DeepB'),
        "msoft":    Hist(50 ,(0,200)    ,'msoft for highest pT jet','Fractional Distribution','netplots/msoft'),
        "DDBvL":    Hist(55 ,(0,1.1)    ,'DDBvL for highest pT jet','Fractional Distribution','netplots/DDBvL'),
        "H4qvs":    Hist(20 ,(0,1)      ,'H4qvs for highest pT jet','Fractional Distribution','netplots/H4qvs'),
        "npvs":     Hist(40 ,(0,80)     ,'npvs per event','Fractional Distribution','netplots/npvs'),
        "npvsG":    Hist(40 ,(0,80)     ,'npvsGood per event','Fractional Distribution','netplots/npvsG')
        #"LHEHT":    Hist(400,(0,4000)   ,'LHE_HT for highest pT jet in passing signal (red), BG (blue), and raw BG (black) events','% Distribution','netplots/LHE_HT'),
    }
    tdict = {}
    prefix = ['BG','SG','RW']
    for plot in vplots:
        size = vplots[plot].size
        bounds = vplots[plot].bounds
        for fix in prefix:
            tdict.update({fix+plot:Hist(size,bounds)})
    vplots.update(tdict)
    del tdict
    
    if (dataflag):
        if dataflag == True:
            cutword = 'signal'
        else:
            cutword = 'background'
        def dataswap(plots):
            for key in plots:
                label = plots[key].xlabel.split(cutword)
                for i in range(1,len(label)):
                    label.insert(2*i-1,'data')
                plots[key].xlabel = ''.join(label)
            return plots
        pplots = dataswap(pplots)
        vplots = dataswap(vplots)
        plots  = dataswap(plots )
                
    
    if isLHE:
        lheplots = {}
        for i in range(nlhe):
            lheplots.update({'dist'+str(i):Hist(50,(0,1),'Normalized MC background classifcation','Fraction of Events','otherplots/LHEdist_'+str(i)),})
            lheplots['dist'+str(i)].title = 'Distrubution for LHE segment '+str(i)
#    for plot in plots:
#        plots[plot].title = files[0]
            

    if LOADMODEL:
        if dataflag == 1:
            pf = 'D'
            for key in vplots:
                vplots[key].ylabel = 'Events'
            for key in pplots:
                pplots[key].ylabel = 'Events'
        elif dataflag == -1:
            pf = 'C'
        else:
            pf = 'S'
        for key in vplots:
            vplots[key].fname = pf+vplots[key].fname
        for key in pplots:
            pplots[key].fname = pf+pplots[key].fname
        for key in ['Distribution','DistributionL']:
            plots[key].fname = pf+plots[key].fname

    ## Create an internal figure for pyplot to write to
    plt.figure(1)
    if isLHE:
        nbg = len(bgfiles)/nlhe
        if float(nbg).is_integer():
            nbg = int(nbg)
        else:
            raise Exception('LHE argument specified, but BG files do not divide evenly into '+str(nlhe))
    else:
        nbg = len(bgfiles)
    nsig = len(sigfiles)
    sigmbg = nbg - nsig
    ## Loop over input files
    for fnum in range(max(nbg,nsig)):
        print('bg',nbg,'sig',nsig)
        
        #####################
        # Loading Variables #
        #####################
        if isLHE:
            print('Opening',sigfiles[fnum],'+ LHE Background')    
        else:
            print('Opening',sigfiles[fnum],'+',bgfiles[fnum])
        
        ## Loop some data if the bg/signal files need to be equalized
        if sigmbg > 0:
            print('Catching up signal')
            sigfiles.append(sigfiles[fnum])
            sigmbg = sigmbg - 1
        elif sigmbg < 0:
            if isLHE:
                print('Catching up background')
                for i in range(nlhe):
                    bgfiles.append(bgfiles[fnum+i])
                    sigmbg = sigmbg + 1
            else:
                print('Catching up background')
                bgfiles.append(bgfiles[fnum])
                sigmbg = sigmbg + 1
        print('diff:',sigmbg)
        ## Open our file and grab the events tree
        if isLHE:
            bgevents = []
            for i in range(nlhe):
                idx = fnum*nlhe + i
                print('Opening ',bgfiles[idx])
                bgevents.append(uproot.open(bgfiles[idx]).get('Events'))
        else:
            bgf = uproot.open(bgfiles[fnum])
            bgevents = bgf.get('Events')
        sigf = uproot.open(sigfiles[fnum])
        sigevents = sigf.get('Events')
        
        def loadjets(jets, events,wname=''):
            jets.eta= pd.DataFrame(events.array('FatJet_eta', executor=executor)).rename(columns=inc)
            jets.phi= pd.DataFrame(events.array('FatJet_phi', executor=executor)).rename(columns=inc)
            jets.pt = pd.DataFrame(events.array('FatJet_pt' , executor=executor)).rename(columns=inc)
            jets.mass=pd.DataFrame(events.array('FatJet_mass', executor=executor)).rename(columns=inc)
            jets.CSVV2 = pd.DataFrame(events.array('FatJet_btagCSVV2', executor=executor)).rename(columns=inc)
            jets.DeepB = pd.DataFrame(events.array('FatJet_btagDeepB', executor=executor)).rename(columns=inc)
            jets.DDBvL = pd.DataFrame(events.array('FatJet_btagDDBvL', executor=executor)).rename(columns=inc)
            jets.msoft = pd.DataFrame(events.array('FatJet_msoftdrop', executor=executor)).rename(columns=inc)
            jets.H4qvs = pd.DataFrame(events.array('FatJet_deepTagMD_H4qvsQCD', executor=executor)).rename(columns=inc)
            jets.event = pd.DataFrame(events.array('event', executor=executor)).rename(columns=inc)
            jets.npvs  = pd.DataFrame(events.array('PV_npvs', executor=executor)).rename(columns=inc)
            jets.npvsG = pd.DataFrame(events.array('PV_npvsGood', executor=executor)).rename(columns=inc)
            jets.extweight = jets.event / jets.event
            if wname != '':
                weights = pickle.load(open('weights/'+wname+'-'+fstrip(DATANAME)+'.p',"rb" ))
                for prop in ['genweights','PUweights','normweights']:
                    #print('jets.extweight[1]')#,jets.extweight[1])    
                    jets.extweight[1] = jets.extweight[1] * weights[prop][1]
            else:
                jets.extweight = jets.event / jets.event
            for j in range(1,jets.pt.shape[1]):
                jets.event[j+1] = jets.event[1]
                jets.npvs[j+1] = jets.npvs[1]
                jets.npvsG[j+1] = jets.npvsG[1]
                #if POSTWEIGHT:
                jets.extweight[j+1] = jets.extweight[1]
            return jets
        
        global DATANAME        
        
        if dataflag == True:
            DATANAME = sigfiles[fnum]
            sigjets = loadjets(PhysObj('sigjets'),sigevents)
        else:
            sigjets = loadjets(PhysObj('sigjets'),sigevents,fstrip(sigfiles[fnum]))
        
        if isLHE:
            bgjets = [PhysObj('300'),PhysObj('500'),PhysObj('700'),PhysObj('1000'),PhysObj('1500'),PhysObj('2000'),PhysObj('inf')]
            for i in range(nlhe):
                bgjets[i] = loadjets(bgjets[i],bgevents[i],fstrip(bgfiles[(fnum*nlhe)+i]))                    
        else:
            if dataflag == -1:
                DATANAME = bgfiles[fnum]
                bgjets = loadjets(PhysObj('bgjets'),bgevents)
            else:
                bgjets = loadjets(PhysObj('bgjets'),bgevents,fstrip(bgfiles[fnum]))

        print(f"Processing {str(len(sigjets.eta))} {Skey} events")
        
        if not dataflag:
            pdgida  = sigevents.array('GenPart_pdgId')
            paridxa = sigevents.array('GenPart_genPartIdxMother')
            parida  = pdgida[paridxa] 
            
            bs = PhysObj('bs')
            
            ## Removes all particles that do not have A parents 
            ## from the GenPart arrays, then removes all particles 
            ## that are not bs after resizing the pdgid array to be a valid mask
            
            bs.oeta = pd.DataFrame(sigevents.array('GenPart_eta')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.ophi = pd.DataFrame(sigevents.array('GenPart_phi')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            bs.opt  = pd.DataFrame(sigevents.array('GenPart_pt' )[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            
            ## Test b order corresponds to As
            testbs = pd.DataFrame(sigevents.array('GenPart_genPartIdxMother')[abs(parida)==Aid][abs(pdgida)[abs(parida)==Aid]==5]).rename(columns=inc)
            ## The first term checks b4 has greater idx than b1, the last two check that the bs are paired
            if ((testbs[4]-testbs[1]).min() <= 0) or ((abs(testbs[2]-testbs[1]) + abs(testbs[4])-testbs[3]).min() != 0):
                print('b to A ordering violated - time to do it the hard way')
                sys.exit()
        
            As = PhysObj('As')    
            As.oeta = pd.DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.ophi = pd.DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.opt =  pd.DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            As.omass =pd.DataFrame(sigevents.array('GenPart_mass', executor=executor)[abs(parida)==25][abs(pdgida)[abs(parida)==25]==Aid]).rename(columns=inc)
            
            higgs = PhysObj('higgs')    
            higgs.eta = pd.DataFrame(sigevents.array('GenPart_eta', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.phi = pd.DataFrame(sigevents.array('GenPart_phi', executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            higgs.pt =  pd.DataFrame(sigevents.array('GenPart_pt' , executor=executor)[abs(parida)!=25][abs(pdgida)[abs(parida)!=25]==25]).rename(columns=inc)
            
            slimjets = PhysObj('slimjets')
            slimjets.eta= pd.DataFrame(sigevents.array('Jet_eta', executor=executor)).rename(columns=inc)
            slimjets.phi= pd.DataFrame(sigevents.array('Jet_phi', executor=executor)).rename(columns=inc)
            slimjets.pt = pd.DataFrame(sigevents.array('Jet_pt' , executor=executor)).rename(columns=inc)
            slimjets.mass=pd.DataFrame(sigevents.array('Jet_mass', executor=executor)).rename(columns=inc)
            #sigjets.CSVV2 = pd.DataFrame(sigevents.array('FatJet_btagCSVV2')).rename(columns=inc)
            slimjets.DeepB = pd.DataFrame(sigevents.array('Jet_btagDeepB', executor=executor)).rename(columns=inc)
            #sigjets.DDBvL = pd.DataFrame(sigevents.array('FatJet_btagDDBvL')).rename(columns=inc)
            #sigjets.msoft = pd.DataFrame(sigevents.array('FatJet_msoftdrop')).rename(columns=inc)
            slimjets.DeepFB= pd.DataFrame(sigevents.array('Jet_btagDeepFlavB', executor=executor)).rename(columns=inc)
            slimjets.puid = pd.DataFrame(sigevents.array('Jet_puId', executor=executor)).rename(columns=inc)
        

            ## Figure out how many bs and jets there are
            nb = bs.oeta.shape[1]
            njet= sigjets.eta.shape[1]
            #nsjet=slimjets.eta.shape[1]
            na = As.oeta.shape[1]
            if na != 2:
                print("More than two As per event, found "+str(na)+", halting")
                sys.exit()
            
            ## Create sorted versions of A values by pt
            for prop in ['eta','phi','pt','mass']:
                As[prop] = pd.DataFrame()
                for i in range(1,3):
                    As[prop][i] = As['o'+prop][As.opt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
                ## Clean up original ordered dataframes; we don't really need them
                #del As['o'+prop]
            
            ## Reorder out b dataframes to match sorted A parents
            tframe = pd.DataFrame()
            tframe[1] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[2] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[1]
            tframe[3] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            tframe[4] = (As.opt.rank(axis=1,ascending=False,method='first')==1)[2]
            for prop in ['eta','phi','pt']:
                bs[prop] = pd.DataFrame()
                bs[prop][1] = bs['o'+prop][tframe][1].dropna().append(bs['o'+prop][tframe][3].dropna()).sort_index()
                bs[prop][2] = bs['o'+prop][tframe][2].dropna().append(bs['o'+prop][tframe][4].dropna()).sort_index()
                bs[prop][3] = bs['o'+prop][~tframe][1].dropna().append(bs['o'+prop][~tframe][3].dropna()).sort_index()
                bs[prop][4] = bs['o'+prop][~tframe][2].dropna().append(bs['o'+prop][~tframe][4].dropna()).sort_index()
                ## Clean up original ordered dataframes; we don't really need them.
                #del bs['o'+prop]
            
#           ## Sort our b dataframes in descending order of pt
#           for prop in ['spt','seta','sphi']:
#               bs[prop] = pd.DataFrame()
#           #bs.spt, bs.seta, bs.sphi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#               for i in range(1,nb+1):
#                   bs[prop][i] = bs[prop[1:]][bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.seta[i] = bs.eta[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
#               #bs.sphi[i] = bs.phi[bs.pt.rank(axis=1,ascending=False,method='first')==i].max(axis=1)
            
#           plots['genAmass'].dfill(As.mass)

            ev = Event(bs,sigjets,As,higgs)
        else:
            ev = Event(sigjets)
            
        
        if isLHE:
            for jets in bgjets+[sigjets]:
                jets.cut(jets.pt > 240)#240)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#90)#0.25)
                #
                jets.cut(jets.mass > 90)
        else:
            for jets in [bgjets, sigjets]:
                jets.cut(jets.pt > 240)#170)
                jets.cut(abs(jets.eta)<2.4)
                jets.cut(jets.DDBvL > 0.8)#0.6)
                jets.cut(jets.DeepB > 0.4184)
                jets.cut(jets.msoft > 90)#0.25)
                #
                jets.cut(jets.mass > 90)

        if not dataflag:
            bs.cut(bs.pt>5)
            bs.cut(abs(bs.eta)<2.4)
            ev.sync()
        
            slimjets.cut(slimjets.DeepB > 0.1241)
            slimjets.cut(slimjets.DeepFB > 0.277)
            slimjets.cut(slimjets.puid > 0)
            slimjets.trimto(jets.eta)
        ## Implement muon cuts for QCD in comparison to data
        elif dataflag == True:
#            smuplots = {
#                "pt":       Hist(50 ,(0,200)  ,'pT for highest pT muon','Fractional Distribution','muplots/pt'),
#                "eta":      Hist(15 ,(0,3)      ,'|eta| for highest pT muon','Fractional Distribution','muplots/eta'),
#                "dxy":      Hist(25 ,(0,0.5)     ,'dxy for highest pT muon','Fractional Distribution','muplots/dxy'),
#                "dxyErr":     Hist(25 ,(0,0.01)   ,'dxyerr for highest pT muon','Fractional Distribution','muplots/dxyerr'),
#                "dxyoErr":    Hist(25 ,(0,100)   ,'dxy / dxyerr for highest pT muon','Fractional Distribution','muplots/dxyoerr'),
#                }
#            bmuplots = cp.deepcopy(smuplots)
            
            muons = PhysObj('Muon',sigfiles[fnum],'mediumId','eta','pt','dxy','dxyErr')
            ev.register(muons)
            muons.cut(muons.mediumId > 0.9)
            muons.cut(abs(muons.eta) < 1.5)
            muons.cut(np.logical_or(
                    np.logical_and(muons.pt > 7,abs(muons.dxy/muons.dxyErr) > 4),
                    np.logical_and(muons.pt > 8,abs(muons.dxy/muons.dxyErr) > 3)))
            ev.sync()
            
#            for prop in ['pt','eta','dxy','dxyErr']:
#                smuplots[prop].dfill(muons[prop][muons.pt.rank(axis=1,method='first') == 1])
#            smuplots['dxyoErr'].dfill(muons['dxy'][muons.pt.rank(axis=1,method='first') == 1] /
#                    muons['dxyErr'][muons.pt.rank(axis=1,method='first') == 1])
#            
#            muons.extweight = pd.DataFrame()
#            muons.extweight[1] = np.sum(sigjets.extweight[sigjets.pt.rank(axis=1,method='first') == 1],axis=1)
            
            ## Apply these cuts to data events as well
            if isLHE:
                bmuons = []
#                bmwtpiece = []
                for i in range(nlhe):
                    idx = fnum*nlhe + i
                    bmuons.append(PhysObj('Muon',bgfiles[idx],'mediumId','eta','pt','dxy','dxyErr'))
                    bmuons[i].cut(bmuons[i].mediumId > 0.9)
                    bmuons[i].cut(abs(bmuons[i].eta) < 1.5)
                    bmuons[i].cut(np.logical_or(
                            np.logical_and(bmuons[i].pt > 7,abs(bmuons[i].dxy/bmuons[i].dxyErr) > 4),
                            np.logical_and(bmuons[i].pt > 8,abs(bmuons[i].dxy/bmuons[i].dxyErr) > 3)))
                    bgjets[i].trimto(bmuons[i].pt)
#                    bmuons[i].trimto(bgjets[i].pt)
#                    bmuons[i].extweight = pd.DataFrame()
#                    bmuons[i].extweight[1] = lheweights[i] * np.sum(bgjets[i].extweight[bgjets[i].pt.rank(axis=1,method='first') == 1],axis=1)
#                    bmwtpiece.append(bmuons[i].extweight[1])
#                    
#                bmweightsum = np.sum(pd.concat(bmwtpiece,ignore_index=True))
#                    
#                for i in range(nlhe):
#                    if bmuons[i].pt.shape[0] > 0:
#                        bmuons[i]['extweight'] = bmuons[i]['extweight'] * np.sum(muons['extweight'])/bmweightsum
#                        for prop in ['pt','eta','dxy','dxyErr']:
#                            bmuplots[prop].fill(np.sum(bmuons[i][prop][bmuons[i].pt.rank(axis=1,method='first') == 1],axis=1),
#                                    weights=bmuons[i].extweight[1])
#                        bmuplots['dxyoErr'].fill(np.sum(bmuons[i].dxy[bmuons[i].pt.rank(axis=1,method='first') == 1],axis=1) / np.sum(bmuons[i].dxyErr[bmuons[i].pt.rank(axis=1,method='first') == 1],axis=1),
#                                weights=bmuons[i].extweight[1])
            else:
                muons = PhysObj('Muon',bgfiles[fnum],'mediumId','eta','pt','dxy','dxyErr')
                muons.cut(muons.mediumId > 0.9)
                muons.cut(abs(muons.eta) < 1.5)
                muons.cut(np.logical_or(
                        np.logical_and(muons.pt > 7,abs(muons.dxy/muons.dxyErr) > 4),
                        np.logical_and(muons.pt > 8,abs(muons.dxy/muons.dxyErr) > 3)))
                bgjets.trimto(muons.pt)
            plt.clf()
#            for prop in smuplots:
#                smuplots[prop].make(linestyle='-',**plargs[Skey])
#                bmuplots[prop].plot(same=True,linestyle='-',legend=['Data','Background'],**plargs[Bkey])
        ## Extra debug table check
#       
#        dbframe = pd.DataFrame()
#        sigjets1 = sigjets.cut(abs(sigjets.eta) > 1.8,split=True)
#        sigjets2 = sigjets.cut(abs(sigjets.eta) > 0.7,split=True)
#        dbframe[0] = [sigjets.eta.shape[0],
#                np.sum(np.sum(sigjets.extweight[sigjets.pt.rank(axis=1,method='first') == 1])),
#                sigjets1.eta.shape[0],
#                np.sum(np.sum(sigjets2.extweight[sigjets2.pt.rank(axis=1,method='first') == 1]))
#                ]
#        for i in range(nlhe):
#            bgjets1 = bgjets[i].cut(abs(bgjets[i].eta) > 1.8,split=True)
#            bgjets2 = bgjets[i].cut(abs(bgjets[i].eta) > 0.7,split=True)
#            dbframe[i+1] = [bgjets[i].eta.shape[0],
#                    np.sum(np.sum(bgjets[i].extweight[bgjets[i]['pt'].rank(axis=1,method='first') == 1])),
#                    bgjets1.eta.shape[0],
#                    np.sum(np.sum(bgjets2.extweight[bgjets2.pt.rank(axis=1,method='first') == 1]))
#                    ]
#        np.savetxt('DEBUGarray.txt',dbframe)
        
        ##############################
        # Processing and Calculation #
        ##############################

        if not dataflag:
            ## Create our dR dataframe by populating its first column and naming it accordingly
            jbdr2 = pd.DataFrame(np.power(sigjets.eta[1]-bs.eta[1],2) + np.power(sigjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            sjbdr2= pd.DataFrame(np.power(slimjets.eta[1]-bs.eta[1],2) + np.power(slimjets.phi[1]-bs.phi[1],2)).rename(columns={1:'Jet 1 b 1'})
            ## Loop over jet x b combinations
            jbstr = []
            for j in range(1,njet+1):
                for b in range(1,nb+1):
                    ## Make our column name
                    jbstr.append("Jet "+str(j)+" b "+str(b))
                    if (j+b==2):
                        continue
                    ## Compute and store the dr of the given b and jet for every event at once
                    jbdr2[jbstr[-1]] = pd.DataFrame(np.power(sigjets.eta[j]-bs.eta[b],2) + np.power(sigjets.phi[j]-bs.phi[b],2))
                    sjbdr2[jbstr[-1]]= pd.DataFrame(np.power(slimjets.eta[j]-bs.eta[b],2) + np.power(slimjets.phi[j]-bs.phi[b],2))
        
            ## Create a copy array to collapse in jets instead of bs
            blist = []
            sblist = []
            for b in range(nb):
                blist.append(np.sqrt(jbdr2.filter(like='b '+str(b+1))))
                blist[b] = blist[b][blist[b].rank(axis=1,method='first') == 1]
                blist[b] = blist[b].rename(columns=lambda x:int(x[4:6]))
                sblist.append(np.sqrt(sjbdr2.filter(like='b '+str(b+1))))
                sblist[b] = sblist[b][sblist[b].rank(axis=1,method='first') == 1]
                sblist[b] = sblist[b].rename(columns=lambda x:int(x[4:6]))
        
            ## Trim resolved jet objects        
#           if resjets==3:
#               for i in range(nb):
#                   for j in range(nb):
#                       if i != j:
#                           blist[i] = blist[i][np.logical_not(blist[i] > blist[j])]
#                           blist[i] = blist[i][blist[i]<0.4]
        
            ## Cut our events to only events with 3-4 bs in one fatjet of dR<0.8
            fjets = blist[0][blist[0]<0.8].fillna(0)/blist[0][blist[0]<0.8].fillna(0)
            for i in range(1,4):
                fjets = fjets + blist[i][blist[i]<0.8].fillna(0)/blist[i][blist[i]<0.8].fillna(0)
            fjets = fjets.max(axis=1)
            fjets = fjets[fjets==4].dropna()
            sigjets.trimto(fjets)
            ev.sync()
        

        
        ##################################
        # Preparing Neural Net Variables #
        ##################################
        bgjetframe = pd.DataFrame()
        extvars = ['event','extweight','npvs','npvsG']
        
        if isLHE:
            bgpieces = []
            wtpieces = []
            
            for i in range(nlhe):
                tempframe = pd.DataFrame()
                twgtframe = pd.DataFrame()
                for prop in netvars+extvars:
                    twgtframe[prop] = bgjets[i][prop][bgjets[i]['pt'].rank(axis=1,method='first') == 1].max(axis=1)
                if 'eta' in netvars:
                    twgtframe['eta'] = abs(twgtframe['eta'])
                twgtframe['val'] = 0
                tempframe = twgtframe.sample(frac=lheweights[i],random_state=6)
                twgtframe['extweight'] = twgtframe['extweight'] * lheweights[i]
                bgpieces.append(tempframe)
                #pickle.dump(tempframe, open(filefix+str(i)+"piece.p", "wb"))
                wtpieces.append(twgtframe)
            bgjetframe = pd.concat(bgpieces,ignore_index=True)
            bgrawframe = pd.concat(wtpieces,ignore_index=True)
            bgjetframe = bgjetframe.dropna()
            bgrawframe = bgrawframe.dropna()
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        else:
            for prop in netvars + extvars:
                bgjetframe[prop] = bgjets[prop][bgjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
            bgjetframe['eta'] = abs(bgjetframe['eta'])
            bgjetframe['val'] = 0
            bgtrnframe = bgjetframe[bgjetframe['event']%2 == 0]
        
        nbg = bgtrnframe.shape[0]
            
        sigjetframe = pd.DataFrame()
        for prop in netvars + extvars:
            sigjetframe[prop] = sigjets[prop][sigjets['pt'].rank(axis=1,method='first') == 1].max(axis=1)
        if 'eta' in netvars:    
            sigjetframe['eta'] = abs(sigjetframe['eta'])
        sigjetframe['val'] = 1
        sigtrnframe = sigjetframe[sigjetframe['event']%2 == 0]
        nsig = sigtrnframe.shape[0]
        
        
        print(f"{Skey} cut to {sigjetframe.shape[0]} events")
        print(f"{Bkey} has {bgjetframe.shape[0]} intended events")
            
        extvars = extvars + ['val']
        #######################
        # Training Neural Net #
        #######################
        
        if TUTOR == True:
            tutor(bgjetframe,sigjetframe)
            sys.exit()
            
        #if not isLHE:
        #    X_test = pd.concat([bgjetframe.drop(bgtrnframe.index), sigjetframe.drop(sigtrnframe.index)])#,ignore_index=True)
        #    X_train = pd.concat([bgtrnframe,sigtrnframe])#,ignore_index=True)
        #    passnum = 0.9       
        #else:
        if LOADMODEL:
            if isLHE:
                bgjetframe=bgrawframe
            ## Normalize event number between QCD and data samples
            if dataflag == True:
                bgjetframe['extweight'] = bgjetframe['extweight'] * np.sum(sigjetframe['extweight'])/np.sum(bgjetframe['extweight'])
                
            X_inputs = pd.concat([bgjetframe,sigjetframe])
            W_inputs = X_inputs['extweight']
            Y_inputs = X_inputs['val']
            X_inputs = X_inputs.drop(extvars,axis=1)
            model = keras.models.load_model('btagfiles/weighted.hdf5', compile=False) #archive
            scaler = pickle.load( open("btagfiles/weightedscaler.p", "rb" ) )
            ##
            #print(scaler.transform(bgpieces[1].drop('val',axis=1)))
            ##
            X_inputs = scaler.transform(X_inputs)
        else:
            X_test = pd.concat([bgjetframe.drop(bgtrnframe.index),sigjetframe.drop(sigtrnframe.index)])
            X_train = pd.concat([bgtrnframe,sigtrnframe])
            #for plot in plots:
            #plots[plot].title = 'Post-Weighted Training'
            W_test = X_test['extweight']
            W_train = X_train['extweight']
                
            Y_test = X_test['val']
            Y_train= X_train['val']
        
            X_test = X_test.drop(extvars,axis=1)
            X_train = X_train.drop(extvars,axis=1)

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            history = model.fit(X_train, Y_train, epochs=epochs, batch_size=5128,shuffle=True,verbose=VERBOSE)

            rocx, rocy, roct = roc_curve(Y_test, model.predict(X_test).ravel())
            trocx, trocy, troct = roc_curve(Y_train, model.predict(X_train).ravel())
            test_loss, test_acc = model.evaluate(X_test, Y_test)
            print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))
    
        passnum = 0.9
        ##################################
        # Analyzing and Plotting Outputs #
        ##################################
        
        if LOADMODEL:
            diststt = model.predict(X_inputs[Y_inputs==1])
            distbtt  = model.predict(X_inputs[Y_inputs==0])
        else:
            diststr = model.predict(X_train[Y_train==1])
            distste = model.predict(X_test[Y_test==1])
            distbtr = model.predict(X_train[Y_train==0])
            distbte = model.predict(X_test[Y_test==0])
            diststt = model.predict(scaler.transform(sigjetframe.drop(extvars,axis=1)))
            distbtt = model.predict(scaler.transform(bgjetframe.drop(extvars,axis=1)))
        
        #if isLHE:
        #    for i in range(nlhe):
        #        piece = wtpieces[i].drop(extvars,axis=1)
        #        piece = piece.reset_index(drop=True)
        #        piece = scaler.transform(piece)
        #        lhedist = model.predict(piece)
        #        #if POSTWEIGHT:
        #        lheplots['dist'+str(i)].fill(lhedist,wtpieces[i]['extweight'])
        #        #else:
        #        #    lheplots['dist'+str(i)].fill(lhedist)
        
        if LOADMODEL:
            plots['DistSte'].fill(diststt,W_inputs[Y_inputs==1])
            plots['DistBte'].fill(distbtt,W_inputs[Y_inputs==0])
            plots['WeightSte'].fill(W_inputs[Y_inputs==1])
            plots['WeightBte'].fill(W_inputs[Y_inputs==0])
        else:
            hist = pd.DataFrame(history.history)
            #for h in history:
                #hist = pd.concat([hist,pd.DataFrame(h.history)],ignore_index=True)
            hist['epoch'] = history.epoch
            plots['LossvEpoch'][0]=hist['loss']
            if 'acc' in hist.columns:
                plots['AccvEpoch'][0]=hist['acc']
            elif 'accuracy' in hist.columns:
                plots['AccvEpoch'][0]=hist['accuracy']
            else:
                plots['AccvEpoch'][0]=hist['loss']
                #plots['LossvEpoch'][0]=hist['epoch']
                #plots['AccvEpoch'][0]=hist['epoch']
            plots['LossvEpoch'].plot()
            plots['AccvEpoch'].plot()
                
            plots['DistStr'].fill(diststr,W_train[Y_train==1])
            plots['DistSte'].fill(distste,W_test[Y_test==1])
            plots['DistBtr'].fill(distbtr,W_train[Y_train==0])
            plots['DistBte'].fill(distbte,W_test[Y_test==0])
            plots['WeightStr'].fill(W_train[Y_train==1])
            plots['WeightSte'].fill(W_test[Y_test==1])
            plots['WeightBtr'].fill(W_train[Y_train==0])
            plots['WeightBte'].fill(W_test[Y_test==0])
            
            plt.clf()
            plt.plot([0,1],[0,1],'k--')
            plt.plot(rocx,rocy,'red')
            plt.plot(trocx,trocy,'b:')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(['y=x','Validation','Training'])
            plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
            plt.savefig(plots['Distribution'].fname+'ROC')
            
        #else:
        #    plots['DistStr'].fill(diststr)
        #    plots['DistSte'].fill(distste)
        #    plots['DistBtr'].fill(distbtr)
        #    plots['DistBte'].fill(distbte)
        #    plt.clf()
            
        for col in netvars + ['npvs','npvsG']:
            vplots['BG'+col].fill(bgjetframe.reset_index(drop=True)[col],bgjetframe.reset_index(drop=True)['extweight'])
            vplots['SG'+col].fill(sigjetframe.reset_index(drop=True)[col],sigjetframe.reset_index(drop=True)['extweight'])

            pplots['SG'+col].fill(sigjetframe.reset_index(drop=True)[col],sigjetframe.reset_index(drop=True)['extweight'])
            pplots['SPS'+col].fill(sigjetframe[diststt > passnum].reset_index(drop=True)[col],sigjetframe[diststt > passnum].reset_index(drop=True)['extweight'])
            pplots['SFL'+col].fill(sigjetframe[diststt <= passnum].reset_index(drop=True)[col],sigjetframe[diststt <= passnum].reset_index(drop=True)['extweight'])
            pplots['BG'+col].fill(bgjetframe.reset_index(drop=True)[col],bgjetframe.reset_index(drop=True)['extweight'])
            pplots['BPS'+col].fill(bgjetframe[distbtt > passnum].reset_index(drop=True)[col],bgjetframe[distbtt > passnum].reset_index(drop=True)['extweight'])
            pplots['BFL'+col].fill(bgjetframe[distbtt <= passnum].reset_index(drop=True)[col],bgjetframe[distbtt <= passnum].reset_index(drop=True)['extweight'])

        

    for p in [plots['DistStr'],plots['DistSte'],plots['DistBtr'],plots['DistBte'],
              plots['WeightStr'],plots['WeightSte'],plots['WeightBtr'],plots['WeightBte']]:
        if sum(p[0]) != 0:
            p.ndivide(sum(p[0]))
        
    if LOADMODEL:
        if dataflag==False:
            leg = ['Signal','Background']
        elif dataflag==True:
            leg = ['Data','Background']
        else:
            leg = ['Signal','Data']
            
        plt.clf()
        plots['DistSte'].make(linestyle='-',**plargs[Skey])
        plots['DistBte'].make(linestyle=':',**plargs[Bkey])
        plots['Distribution'].plot(same=True,legend=leg)
        plt.clf()
        plots['DistSte'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['DistBte'].make(linestyle=':',logv=True,**plargs[Bkey])
        plots['DistributionL'].plot(same=True,logv=True,legend=leg)
    else:
        leg = ['Signal (training)','Background (training)','Signal (testing)', 'Background(testing)']
            
        plt.clf()
        plots['DistStr'].make(linestyle='-',**plargs[Skey])
        plots['DistBtr'].make(linestyle='-',**plargs[Bkey])
        plots['DistSte'].make(linestyle=':',**plargs[Skey])
        plots['DistBte'].make(linestyle=':',**plargs[Bkey])
        plots['Distribution'].plot(same=True,legend=leg)
    
        plt.clf()
        plots['DistStr'].make(linestyle='-',logv=True,**plargs[Skey])
        plots['DistBtr'].make(linestyle='-',logv=True,**plargs[Bkey])
        plots['DistSte'].make(linestyle=':',logv=True,**plargs[Skey])
        plots['DistBte'].make(linestyle=':',logv=True,**plargs[Bkey])
        plots['DistributionL'].plot(same=True,logv=True,legend=leg)
    
    #if POSTWEIGHT:
    if LOADMODEL:
        plt.clf()
        plots['WeightSte'].make(linestyle='-',**plargs[Skey])
        plots['WeightBte'].make(linestyle=':',**plargs[Bkey])
        plots['Weights'].plot(same=True,legend=leg)
    else:
        plt.clf()
        plots['WeightStr'].make(color='red',linestyle='-',htype='step')
        plots['WeightBtr'].make(color='blue',linestyle='-',htype='step')
        plots['WeightSte'].make(color='red',linestyle=':',htype='step')
        plots['WeightBte'].make(color='blue',linestyle=':',htype='step')
        plots['Weights'].plot(same=True,legend=leg)
    
    if dataflag != True:
        for col in netvars:
            for plot in vplots:
                vplots[plot].ndivide(sum(abs(vplots[plot][0]+.0001)))
            for plot in pplots:
                pplots[plot].ndivide(sum(abs(pplots[plot][0]+.0001)))

    if isLHE:
        for i in range(nlhe):
            lheplots['dist'+str(i)][0] = lheplots['dist'+str(i)][0]/sum(lheplots['dist'+str(i)][0])
            lheplots['dist'+str(i)].plot(htype='step')#,logv=True)
        
    for col in netvars + ['npvs','npvsG']:
        plt.clf()
        vplots['SG'+col].make(linestyle='-',**plargs[Skey])
        vplots['BG'+col].make(linestyle=':',**plargs[Bkey],error=dataflag)
        #if POSTWEIGHT:
        vplots[col].title = 'With post-weighted network training'
        #else:
        #   vplots[col].title = 'With weighted network training'
        vplots[col].plot(same=True,legend=[Skey,Bkey])
        
        plt.clf()
        pplots['SG'+col].make(color='red'  ,linestyle='-',htype='step')
        pplots['SFL'+col].make(color='black',linestyle='--',htype='step')
        pplots['SPS'+col].make(color='blue' ,linestyle=':',htype='step')
        pplots[col].plot(same=True,legend=[f"All {Skey}", f"Failing {Skey}",f"Passing {Skey}"])
        
        plt.clf()
        pplots['BG'+col].make(color='red'  ,linestyle='-',htype='step')
        pplots['BFL'+col].make(color='black',linestyle='--',htype='step')
        pplots['BPS'+col].make(color='blue' ,linestyle=':',htype='step')
        pplots['B'+col].plot(same=True,legend=[f"All {Bkey}",f"Failing {Bkey}",f"Passing {Bkey}"])
        
        
    #if POSTWEIGHT:
    #model.save('postweighted.hdf5')
    #pickle.dump(scaler, open("postweightedscaler.p", "wb"))
    #else:
    model.save('weighted.hdf5')
    pickle.dump(scaler, open("weightedscaler.p", "wb"))
    #pickle.dump(sigjetframe, open("sigj.p","wb"))
    
        
    #%%
    #return auc(rocx,rocy)
        #sys.exit()

## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        nrgs = len(sys.argv)
        sigfiles = []
        bgfiles = []
        datafiles = []
        isLHE=False
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
            elif '-LHE' in arg:
                isLHE = True
        #print('-')
        #print('sigfiles',sigfiles,'datafiles',datafiles)
        #print('-')
        if not len(datafiles):
            dataflag = False
        elif len(sigfiles) and not len(bgfiles):
            bgfiles = datafiles
            dataflag = -1
        elif len(bgfiles) and not len(sigfiles):
            sigfiles = datafiles
            dataflag = True
        else: dialogue()
        
        ana(sigfiles,bgfiles,isLHE,dataflag)

    else:
        dialogue()
        
def dialogue():
    print("Expected:\n mndwrm.py [-LHE] <-f/-l>s (signal.root) <-f/-l>b (background.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    print("s      Marks the following file(s) as signal")
    print("b      Marks the following file(s) as background")
    print("d      Marks the following file(s) as data - used to substitute 's' or 'd' as only 2 input types are expected")
    print("-LHE   Indicates background files are split by LHE, and should be merged")
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
