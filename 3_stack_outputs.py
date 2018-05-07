# -*- coding: utf-8 -*-

import pandas as pd
import sys
from os import listdir
from os.path import isfile, join

nrxn = sys.argv[1]
output = sys.argv[2]

results = pd.DataFrame(columns=['idx','reaction','rxn_id','regulator','equation',\
                    'meas_flux','pred_flux','best_fit','best_lik','lik_cond'])
for idx in range(int(nrxn)):
    onlyfiles = [f for f in listdir(str('results%s'%idx)) if isfile(join(str('results%s'%idx), f))]
    for i in onlyfiles:
        if i!='reaction_forms.pickle':
            add = pd.read_pickle(str('results%s/%s' % (idx,i)))
            results = results.append([add])

results = results.sort_values(by='best_lik')
results.reset_index(drop=True,inplace=True)

results.to_pickle(output)
