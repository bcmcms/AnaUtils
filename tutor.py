#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Controls the targeted network to train
from mndwrm import ana as task

def teach(sigfiles, bgfiles):
    records = {}
    rsums = {}
    for l1 in [6,8,10,12,14]:
        for l2 in [1,2,3,4,5,6,7]:
            for l3 in [1,2,3,4,5]:
                rname = str(l1)+' '+str(l2)+' '+str(l3)
                aoc = []
                for i in range(10):
                    aoc[i] = (task(sigfiles,bgfiles,l1,l2,l3))
                    
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
                    
                


## Define 'main' function as primary executable
def main():
    if (len(sys.argv) > 1):
        sigfiles = []
        bgfiles = []
        ## Check for file sources
        if '-f' in sys.argv:
            idx = sys.argv.index('-f')+1
            try:                
                for i in sys.argv[idx:]:
                    if i == '-s':
                        fileptr = sigfiles
                    elif i == '-b':
                        fileptr = bgfiles
                    else:
                        fileptr.append(i)
            except:
                dialogue()
        elif '-l' in sys.argv:
            idx = sys.argv.index('l')+1
            try:
                for i in sys.argv[idx:]:
                    if i == '-s':
                        fileptr = sigfiles
                    elif i == '-b':
                        fileptr = bgfiles
                    else:
                        with open(sys.argv[3],'r') as rfile:
                            for line in rfile:
                                fileptr.append(line.strip('\n')) 
            except:
                dialogue()
        else:
            dialogue()
        ## Check specified run mode
        teach(sigfiles,bgfiles)
    else:
        dialogue()
        
def dialogue():
    print("Expected tutor.py <-f/-l> -s (signal.root) -b (background.root)")
    print("---formatting flags--")
    print("-f     Targets a specific file to run over")
    print("-l     Specifies a list containing all files to run over")
    sys.exit(0)
    
if __name__ == "__main__":
    main()

