import pandas as pd
import numpy as np
import sys, getopt
sys.path.append('../')
from Modules import thesis_module as tm

infile = "../randomSample_purified_data2.csv"

try:
    opts, args = getopt.getopt(argv,"i",["infile="])
except getopt.GetoptError:
    sys.exit(2)
if '?' in args or 'help' in args:
    print('Help for "processing.py"')
    print('This file is used to process data.')
    print()
    print('Options:')
    print('-i, --infile:  Defines the file that requires processing.')
    print()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i","--infile"):
        infile = arg

df = pd.read_csv(infile, sep = ';')
result = tm.preprocess(infile)
result = result.drop(['platform'], axis=1)

result.to_csv(infile[:-4]+' processed.csv')
