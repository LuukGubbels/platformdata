import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from Modules import thesis_module as tm

infile = "../randomSample_purified_data2.csv"

df = pd.read_csv(infile, sep = ';')
result = tm.preprocess(infile)
result = result.drop(['platform'], axis=1)

result.to_csv(infile[:-4]+' processed.csv')