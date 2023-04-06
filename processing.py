import pandas as pd
import numpy as np
import sys, getopt
sys.path.append('../')
import thesis_module as tm

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"i:o",["infile=","ofile="])
except getopt.GetoptError:
    raise getopt.GetoptError
    sys.exit(2)
if '?' in args or 'help' in args:
    print('Help for "processing.py"')
    print('This file is used to process data.')
    print()
    print('Options:')
    print('-i, --infile:  Defines the file that requires processing. Input as a .csv file with extension. No default.')
    print('-o, --ofile:   Defines the file that is used for output. Input as a .csv file with extension. No default.')
    print()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i","--infile"):
        infile = arg.strip('][').split(',')
    elif opt in ("-o","--ofile"):
        outfile = arg

frames = []
for i in infile:
    df = pd.read_csv(i, sep = ';')
    df = df.fillna(" ")
    df = df[df["text"].str.split().apply(len)>=10]
    frames.append(df)
df = pd.concat(frames, sort=True)
df["text"].str.findall('w{\3,}').str.join(' ')
df["text"] = df["text"].str.replace("  ", " ")
df =  df.drop(["platform"], axis=1)

df.to_csv(outfile+' processed.csv')
