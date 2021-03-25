#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility n00dle.py                               ###
###                                                                  ###
### Currently doesn't support options... but we're improving!        ###
########################################################################

#import os
#import subprocess
#import sys
import uproot, json
import numpy as np
#import awkward
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
#import itertools as it
import copy as cp
from munch import DefaultMunch
# from ROOT import TH1F
#import types
import mplhep as hep
from scipy.stats import norm

def stepx(xs):
    return np.tile(xs, (2,1)).T.flatten()[1:-1]
def stepy(ys):
    return np.tile(ys, (2,1)).T.flatten()

def dphi(phi1,phi2):
    return abs(phi1-phi2).combine(abs(phi1-phi2+(2*np.pi)),min).combine(abs(phi1-phi2-(2*np.pi)),min)

#class HackyTH1(uproot_methods.classes.TH1.Methods, list):
#    def __init__(self, low, high, values, title=""):
#        self._fXaxis = types.SimpleNamespace()
#        self._fXaxis._fNbins = len(values)
#        self._fXaxis._fXmin = low
#        self._fXaxis._fXmax = high
#        for x in values:
#            self.append(float(x))
#        self._fTitle = title
#        self._classname = "TH1F"

class Hist(object):
    def __init__(s,size,bounds,xlabel='',ylabel='',fname='',title=''):
        s.size = size
        s.bounds = bounds
        s.hs = [plt.hist([],size,bounds)[0],plt.hist([],size,bounds)[1]]
        s.xlabel = xlabel
        s.ylabel = ylabel
        s.title = title
        s.fname = fname
        s.fig = ''
        s.ser = s.hs[0]

    def __getitem__(s,i):
        if (i > 1) or (i < -2):
            raise Exception('histo object was accessed with an invalid index')
        return s.hs[i]
    
    def __setitem__(s,i,val):
        if (i > 1) or (i < -2):
            raise Exception('histo object was accessed with an invalid index')
        s.hs[i] = val
        return s

    ## Adds the values of a passed histogram to the class's plot
    def add(s,inplot,split=False):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        if split == True:
            s = cp.deepcopy(s)
        s.hs[0] = s.hs[0] + inplot[0]
        s.ser = s.ser + inplot.ser
        return s
    
    ## Subtracts the values of a passed histogram from the class's plot
    def sub(s,inplot,split=False):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        if split == True:
            s = cp.deepcopy(s) 
        s.hs[0] = s.hs[0] - inplot[0]
        s.ser = s.ser + inplot.ser
        return s

    ## Fills the stored histogram with the supplied values, and tracks squared uncertainty sum
    def fill(s,vals,weights=None):
        vals[vals < s.hs[1][0]] = s.hs[1][0]
        vals[vals > s.hs[1][-1]] = s.hs[1][-1]
        if weights is None:
            s.ser = s.ser + plt.hist(vals,s.hs[1])[0]
        else:
            s.ser = s.ser + plt.hist(vals,s.hs[1],weights=(weights*weights))[0]
        s.hs[0] = s.hs[0] + plt.hist(vals,s.hs[1],weights=weights)[0]
        return s

    ## Fills the stored histogram with values from the supplied dataframe
    def dfill(s,frame):
        s.fill(frame.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        return s

    ## Divides the stored histogram by another, and either changes itself or returns a changed object
    ## Enabling trimnoise attempts to cut out the weird floating point errors you sometimes get when a number isn't exactly 0
    def divideby(s,inplot,split=False,trimnoise=0,errmethod='default'):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        if split:
            s = cp.deepcopy(s)
        inplot = cp.deepcopy(inplot)
        
        if trimnoise:
            s.hs[0][s.hs[0]<trimnoise]=np.nan
            inplot[0][inplot[0]<trimnoise]=np.nan
        
        upper, lower = [],[]
        for i in range(len(s.hs[0])):
            if errmethod == 'default':
                A = s.hs[0][i]
                eA = np.sqrt(s.ser[i])
                B = inplot[0][i]
                eB = np.sqrt(inplot.ser[i])
                s.ser[i] = np.power(A*B,2) * (np.power(eA/A,2) + np.power(eB/B,2))
                s.ser[np.isnan(s.ser)] = 0.
            ## Using normal efficiency calculation changes the format of the error,
            ## So there shouldn't be further histogram value operations done between it and plotting
            elif errmethod == 'effnorm':
                level = 0.68
                total = inplot[0][i]
                if total == 0:
                    lower.append(0)
                    upper.append(0)
                    continue
                avgwgt = inplot.ser[i] / total
                jitter = avgwgt/total
                passed = s.hs[0][i]
                alpha = (1 - level)/2
                avg = passed / total
                sigma = np.sqrt((avgwgt * (avg + jitter) * (1 + jitter - avg))/total)
                delta = norm.ppf(1-alpha,0,sigma)
                upper.append(min(delta*delta,np.power(1-avg,2)))
                lower.append(min(delta*delta,np.power(avg,2)))
        if errmethod == 'effnorm': s.ser = np.array([lower,upper])
        
        # A = s.hs[0]
        # eA = np.sqrt(s.ser)
        s.hs[0] = np.divide(s.hs[0],inplot[0], where=inplot[0]!=0)
        ## Empty bins should have a weight of 0
        s.hs[0][np.isnan(s.hs[0])] = 0
        inplot[0][np.isnan(inplot[0])] = 0
        # ## ex^2 = x^2 * ((eA/A)^2 + (eB/B)^2)
        # s.ser = (s.hs[0]*s.hs[0])*(np.power(eA/A,2)+np.power(np.sqrt(inplot.ser)/inplot[0],2))
        #s.ser[np.isnan(s.ser)] = 0.
        return s
    
    def divideBy(s,*args,**kwargs):
        return s.divideby(*args,**kwargs)
    
    ## Normalizes the histogram by the magnitude of a specified bin
    def norm(s,tar=0,split=False):
        if split:
            s = cp.deepcopy(s)
        nval = s.hs[0][tar]
        s.hs[0] = s.hs[0]/nval
        return s
    
    ## Divides the histogram's bins, and its squared uncertainty sum, by a number.
    def ndivide(s,num=1):
        if num == 0:
            raise Exception(f"You tried to divide {s.fname} by 0")
        s.hs[0] = s.hs[0]/num
        s.ser = s.ser/(num*num)
        return s

    ## Creates and returns a pyplot-compatible histogram object
    def make(s,logv=False,htype='bar',color=None,linestyle='solid',error=False,parent=plt):
        if htype=='err':
            if not color:
                color = 'k'
            binwidth = s.hs[1][2]-s.hs[1][1]
            parent.hlines(s.hs[0],s.hs[1][0:-1],s.hs[1][1:],colors=color)
            plot = parent.errorbar(s.hs[1][:-1]+binwidth/2,s.hs[0],yerr=np.sqrt(s.ser),fmt=f".{color}",
                        color=color,linewidth=2,capsize=3)
            if logv:
                parent.yscale('log')
            return plot
        plot = parent.hist(s.hs[1][:-1],s.hs[1],weights=s.hs[0],
                        log=logv,histtype=htype,color=color,linestyle=linestyle,linewidth=0)
        if error==True:
            parent.fill_between(stepx(s.hs[1]),stepy(s.hs[0]-np.sqrt(s.ser)),stepy(s.hs[0]+np.sqrt(s.ser)),
                             alpha=0.0,hatch='xxxxxx',zorder=2,label='_nolegend_')
            
        return plot
        #return hep.histplot(s.hs[0],s.hs[0],log=logv,histtype=htype,color=color,linestyle=linestyle)
    def plot(s,ylim=False,same=False,legend=False,figure=False,clean=False,**args):
        if not same:
            plt.clf()
        s.make(**args)
        
       
        if not figure:
            fig = plt.gcf()
        else: fig = figure
        fig.set_size_inches(10.0, 6.0)
        
       
        
        if 'parent' in args:
            args['parent'].grid(True)
            if not clean: hep.cms.label(loc=0,year='2018',ax=args['parent'])
            if legend:
                args['parent'].legend(legend,loc=0)
            if ylim:
                args['parent'].set_ylim(ylim)
            if s.xlabel != '':
                args['parent'].set_xlabel(s.xlabel,fontsize=14)
            if s.ylabel != '':
                args['parent'].set_ylabel(s.ylabel,fontsize=18)
        else:    
            plt.grid(True)
            hep.cms.label(loc=0,year='2018')
            if legend:
                 plt.legend(legend,loc=0)
            if ylim:
                plt.ylim(ylim)
            if s.xlabel != '':
                plt.xlabel(s.xlabel,fontsize=14)
            if s.ylabel != '':
                plt.ylabel(s.ylabel,fontsize=18)
            if s.title != '':
                plt.title(s.title)
        if s.fname != '':
            if figure:  figure.savefig(s.fname)
            else:       plt.savefig(s.fname)
        plt.close(s.fig)
       
    ## Shortcut for creating stacked plots of two comperable datasets
    def stackplot(s,phist,ylim=False):
        plt.clf()
        s.make(htype='step',color='black')
        phist.make(htype='step',color='red')
        if ylim:
            plt.ylim(ylim)
        if s.xlabel != '':
            plt.xlabel(s.xlabel)
        if s.ylabel != '':
            plt.ylabel(s.ylabel)
        if s.title != '':
            plt.title(s.title)
        if s.fname != '':
            plt.savefig(s.fname+'_v')    
            
    # def toTH1(s,title,scale=1):
    #     th1 = TH1F(title,title,len(s.hs[0]),s.hs[1][0],s.hs[1][-1])
    #     for i in range(len(s.hs[0])):
    #         th1.SetBinContent(i,s.hs[0][i]*scale)
    #         th1.SetBinError(i,np.sqrt(s.ser[i])*scale)
    #     return th1
    
#    def errtoTH1(s,scale=1):
#        return np.histogram(s.hs[1][12:-5],s.hs[1][12:-4],weights=np.sqrt(s.ser[12:-4])*scale)
#    
##    errToTh1 = errtoTH1
#    def errToTH1(s,scale=1):
#        return s.errtoTH1(scale)


class Hist2d(object):
    def __init__(s,sizes,bounds,xlabel='',ylabel='',fname='',title=''):
        s.sizes = sizes
        s.bounds = bounds
        s.hs = [plt.hist2d([],[],sizes,bounds)[0],plt.hist2d([],[],sizes,bounds)[1],plt.hist2d([],[],sizes,bounds)[2]]#,plt.hist2d([],[],sizes,bounds)]
        s.xlabel = xlabel
        s.ylabel = ylabel
        s.title = title
        s.fname = fname

    def __getitem__(s,i):
        if (i > 2) or (i < -3):
            raise Exception('hist2d object was accessed with an invalid index')
        return s.hs[i]
    
    def __setitem__(s,i,val):
        if (i > 2) or (i < -3):
            raise Exception('hist2d object was accessed with an invalid index')
        s.hs[i] = val
        return s

    def add(s,inplot):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])) or (len(inplot[2]) != len(s.hs[2])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        s.hs[0] = s.hs[0] + inplot[0]
        return s

    def fill(s,valx,valy,weights=None):
        s.hs[0] = s.hs[0] + plt.hist2d(valx,valy,s.sizes,s.bounds,weights=weights)[0]
        return s

    def dfill(s,framex,framey):
        s.fill(framex.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0],\
        framey.melt(value_name=0).drop('variable',axis=1).dropna().reset_index(drop=True)[0])
        return s

    def norm(s,tar=[0,0],split=False):
        if split:
            s = cp.deepcopy(s)
        nval = s.hs[0][tar[0]][tar[1]]
        s.hs[0] = s.hs[0]/nval
        return s

    def make(s,edgecolor='face',linewidth=1):
        plt.clf()
        #out = plt.imshow(s.hs[0].T[::-1],extent=(s.bounds[0][0],s.bounds[0][1],s.bounds[1][0],s.bounds[1][1]),aspect='auto',origin='upper')
        out = plt.pcolor(s.hs[1],s.hs[2],s.hs[0].T,edgecolor=edgecolor,linewidth=linewidth)
        return out    

    def plot(s,logv=False,text=False,empty=False,*args,**kwargs):
        if not empty:
            s.make(*args,**kwargs)
        #print(s.hs[0])
        #print(s.hs[1])
        #print(s.hs[2])
        if text:
            strarray = s.hs[0].round(3).astype(str)
            for i in range(len(s.hs[1])-1):
                for j in range(len(s.hs[2])-1):
                    plt.text(s.hs[1][i]+0.5,s.hs[2][j]+0.5, strarray[i,j],color="w", ha="center", va="center", fontweight='normal',fontsize=9).set_path_effects([PathEffects.withStroke(linewidth=2,foreground='k')])
        else:
            plt.colorbar()
        if s.xlabel != '':
            plt.xlabel(s.xlabel)
        if s.ylabel != '':
            plt.ylabel(s.ylabel)
        if s.title != '':
            plt.title(s.title)
        if s.fname != '':
            plt.savefig(s.fname)


def inc(var):
    return var+1

def fstrip(path):
    return path.split('/')[-1].split('.root')[0]

class PhysObj(DefaultMunch):
    def __init__(s,name='',rfile='',*args,varname=False):
        if len(args) != 0:
            events = uproot.open(rfile).get('Events')
            if not varname:
                varname = name
            for arg in args:
                s[arg] = pd.DataFrame(events.array(varname+'_'+arg)).rename(columns=inc)
        super().__init__(name)

    def __setitem__(s,key,value):
        if not isinstance(value,pd.DataFrame):
            raise Exception("PhysObj elements can only be dataframe objects")
        super().__setitem__(key,value)
        #s.update({key:value})
        return s

    ## Removes events that are missing, in the passed frame
    def trimto(s,frame,split=False):
        if split:
            s = s.deepcopy()
        for elem in s:
            s[elem] = s[elem].loc[frame.index.intersection(s[elem].index)]
        return s
    
    def trimTo(s,*args,**kwargs):
        s = s.trimto(*args,**kwargs)
        return s
    
    ## Removes events that are missing, from the passed frame (probably not ideal to have to do this)
    def trim(s,frame):
        for elem in s:
            frame = frame.loc[s[elem].index.intersection(frame.index)]
        return frame

    ## Removes particles that fail the passed test, and events if they become empty
    def cut(s,mask,split=False):
        if split:
            s = s.deepcopy()
        for elem in s:
            s[elem] = s[elem][mask].dropna(how='all')
        return s
    

class Event():
    def __init__(s,*args):
        if len(args) == 0:
            raise Exception("Event was initialized without an appropriate object")
        s.objs = {}
        for arg in args:
            s.register(arg)
        s.frame = args[0][list(args[0].keys())[0]]

    def __getitem__(s,obj):
        return s.objs[obj]

    def __iter__(s):
        return iter(s.obj)

    ## Adds a new PhysObj to the Event (or replaces an existing one)
    def register(s,obj):
        if not isinstance(obj,PhysObj):
            raise Exception("Non PhysObj object passed to event.")
        s.objs.update({obj.name:obj})
        return s

    ## Looks through all associated objects for disqualified events
    def scan(s):
        for obj in s.objs:
            for elem in s[obj]:
                s.frame = s.frame.loc[s[obj][elem].index.intersection(s.frame.index)]
        return s

    ## Applies disqualified events to all associated objects
    def applycuts(s,split=False):
        ## event splitting is currently broken; applying deepcopy to a physics object attempts to call a string
        if split:
            s = cp.deepcopy(s)
        for obj in s.objs:
            s[obj].trimto(s.frame)
        return s
    
    def sync(s,split=False):
        if split:
            s = cp.deepcopy(s)
        s.scan()
        s.applycuts()
        return s
    
class InputConfig(object):
    def __init__(s,sigfile,bgfile):
        with open(sigfile) as f:
            sigdata = json.load(f)
            if 'isdata' in sigdata:
                s.sigdata =     sigdata['isdata']
            if 'islhe' in sigdata:
                s.siglhe =      sigdata['islhe']
            if 'name' in sigdata:
                s.signame =     sigdata['name']
            if 'normweight' in sigdata:
                snormweight = sigdata['normweight']
            else: snormweight = False
            if 'files' in sigdata:
                s.sigfiles =    sigdata['files']
                s.sigweight =   sigdata['weight']
            elif 'filepairs' in sigdata:
                s.sigfiles,s.sigweight = s.expandpairs(sigdata['filepairs'])
            else: raise NameError("Could not find 'files' or 'filepairs' in input file")
        with open(bgfile) as f:
            bgdata = json.load(f)
            if 'isdata' in bgdata:
                s.bgdata =      bgdata['isdata']
            if 'islhe' in bgdata:
                s.bglhe =       bgdata['islhe']
            if 'name' in bgdata:
                s.bgname =      bgdata['name']
            if 'normweight' in bgdata:
                bnormweight = bgdata['normweight']
            else: bnormweight = False
            if 'files' in bgdata:
                s.bgfiles =     bgdata['files']
                s.bgweight =    bgdata['weight']
            elif 'filepairs' in bgdata:
                s.bgfiles,s.bgweight = s.expandpairs(bgdata['filepairs'])
            else: raise NameError("Could not find 'files' or 'filepairs' in input file")
        if type(s.sigfiles) == str:
            s.sigfiles = [s.sigfiles]
        if type(s.bgfiles) == str:
            s.bgfiles = [s.bgfiles]
        signum = len(s.sigfiles)
        bgnum =  len(s.bgfiles)
        
    
        if s.bglhe == True and s.siglhe == True:
            raise ValueError("Signal and background sources can't both be lhe split")
        elif s.bglhe == True:
            s.size = signum
        elif s.siglhe == True:
            raise ValueError("Currently, signal lhe support is still pending")
        else:
            ## Loop one input file until it reaches the dimensionality of the largest list
            while len(s.sigfiles) < max(signum,bgnum):
                s.sigfiles.append(s.sigfiles[signum*-1])
            while len(s.bgfiles) < max(signum,bgnum):
                s.bgfiles.append(s.bgfiles[signum*-1])  
            s.size = max(signum,bgnum)
        
        ## Loop weights to match input dimensions
        if type(s.sigweight)==list:
            while len(s.sigweight) < len(s.sigfiles):
                s.sigweight.append(s.sigweight[signum*-1])
            if len(s.sigweight) > len(s.sigfiles):
                raise ValueError('Input weight array too large')
        elif type(s.sigweight)==int or type(s.sigweight)==float:
            tlst = []
            for i in range(s.size):
                tlst.append(s.sigweight)
            s.sigweight = tlst
            
        if type(s.bgweight)==list:
            while len(s.bgweight) < len(s.bgfiles):
                s.bgweight.append(s.bgweight[bgnum*-1])
            if len(s.bgweight) > len(s.bgfiles):
                raise ValueError(f"Input weight array too large")
        elif type(s.bgweight)==int or type(s.bgweight)==float:
            tlst = []
            for i in range(s.size):
                tlst.append(s.bgweight)
            s.bgweight=tlst
            
        if snormweight:
            s.sigweight = np.divide(s.sigweight,max(s.sigweight))
        if bnormweight:
            s.bgweight = np.divide(s.bgweight,max(s.bgweight))
            
    def expandpairs(s,pairs):
        flist, wlist = []
        if (len(pairs) % 2):
            raise ValueError("Pair list was found to have odd number of elements during unwrapping")
        for i in range(0,len(pairs),2):
            flist.append(pairs[i])
            wlist.append(pairs[i+1])
        return flist, wlist       
    
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
        
            