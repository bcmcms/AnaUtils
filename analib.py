#! /usr/bin/env python

########################################################################
### NanoAOD analyzer utility n00dle.py                               ###
###                                                                  ###
### Currently doesn't support options... but we're improving!        ###
########################################################################

#import os
#import subprocess
#import sys
import uproot
import numpy as np
#import awkward
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd
#import itertools as it
import copy as cp
from munch import DefaultMunch

class Hist(object):
    def __init__(s,size,bounds,xlabel='',ylabel='',fname='',title=''):
        s.size = size
        s.bounds = bounds
        s.hs = [plt.hist([],size,bounds)[0],plt.hist([],size,bounds)[1]]
        s.xlabel = xlabel
        s.ylabel = ylabel
        s.title = title
        s.fname = fname

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

    def norm(s,tar=0,split=False):
        if split:
            s = cp.deepcopy(s)
        nval = s.hs[0][tar]
        s.hs[0] = s.hs[0]/nval
        return s

    ## Creates and returns a pyplot-compatible histogram object
    def make(s,logv=False,htype='bar',color=None,linestyle='solid'):
        #print(s.hs)
        return plt.hist(s.hs[1][:-1],s.size,s.bounds,weights=s.hs[0],log=logv,histtype=htype,color=color,linestyle=linestyle)

    def plot(s,logv=False,ylim=False,same=False,htype='bar'):
        if not same:
            plt.clf()
        s.make(logv,htype)
        if ylim:
            plt.ylim(ylim)
        if s.xlabel != '':
            plt.xlabel(s.xlabel)
        if s.ylabel != '':
            plt.ylabel(s.ylabel)
        if s.title != '':
            plt.title(s.title)
        if s.fname != '':
            plt.savefig(s.fname)
            
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


class Hist2d(object):
    def __init__(s,sizes,bounds,xlabel='',ylabel='',fname='',title=''):
        s.sizes = sizes
        s.bounds = bounds
        s.hs = [plt.hist2d([],[],sizes,bounds)[0],plt.hist2d([],[],sizes,bounds)[1],plt.hist2d([],[],sizes,bounds)[2],plt.hist2d([],[],sizes,bounds)]
        s.xlabel = xlabel
        s.ylabel = ylabel
        s.title = title
        s.fname = fname

    def __getitem__(s,i):
        if (i > 2) or (i < -3):
            raise Exception('histo object was accessed with an invalid index')
        return s.hs[i]

    def add(s,inplot):
        if (len(inplot[0]) != len(s.hs[0])) or (len(inplot[1]) != len(s.hs[1])) or (len(inplot[2]) != len(s.hs[2])):
            raise Exception('Mismatch between passed and stored histogram dimensions')
        s.hs[0] = s.hs[0] + inplot[0]
        return s

    def fill(s,valx,valy):
        s.hs[0] = s.hs[0] + plt.hist2d(valx,valy,s.sizes,s.bounds)[0]
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

    def plot(s,logv=False,text=False,edgecolor='face',linewidth=1):
        s.make(edgecolor,linewidth,same)
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


class PhysObj(DefaultMunch):
    def __init__(s,name='',rfile='',*args):
        if len(args) != 0:
            events = uproot.open(rfile).get('Events')
            for arg in args:
                s[arg] = pd.DataFrame(events.array(name+'_'+arg)).rename(columns=inc)
        super().__init__(name)

    def __setitem__(s,key,value):
        if not isinstance(value,pd.DataFrame):
            raise Exception("PhysObj elements can only be dataframe objects")
        super().__setitem__(key,value)
        #s.update({key:value})
        return s

    ## Removes events that are missing in the passed frame
    def trimTo(s,frame,split=False):
        if split:
            s = s.copy()
        for elem in s:
            s[elem] = s[elem].loc[frame.index.intersection(s[elem].index)]
        return s
    
    ## Removes events that are missing from the passed frame (probably not ideal to have to do this)
    def trim(s,frame):
        for elem in s:
            frame = frame.loc[s[elem].index.intersection(frame.index)]
        return frame

    ## Removes particles that fail the passed test, and events if they become empty
    def cut(s,mask,split=False):
        if split:
            s = s.copy()
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
        if split:
            s = cp.deepcopy(s)
        for obj in s.objs:
            s[obj].trimTo(s.frame)
        return s

    def sync(s,split=False):
        if split:
            s = cp.deepcopy(s)
        s.scan()
        s.applycuts()
        return s
