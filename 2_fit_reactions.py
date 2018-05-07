# -*- coding: utf-8 -*-

import sys
import pandas as pd
import cobra
import alloregfit as arf

model_path = sys.argv[1]

model = cobra.io.load_matlab_model(model_path)
summary = pd.read_pickle('summary.pickle')
candidates = pd.read_pickle('candidates.pickle')

file = open('run_fit_reaction.txt', 'w')
count = 0
ct = 0
add = ''
for idx in list(summary.index):
    expr,parameters,regulator = arf.write_rate_equations(idx,summary,model,candidates)
    file.write(str('mkdir results%i\n'%idx))
    for i in range(len(expr)):
        add += str('python fit_reaction.py %i %i %s; '%(idx,i,model_path)); ct +=1
        if ct%150 == 0:
            file.write(str('bsub -R "rusage[mem=4096]" -W 24:00 "%s"\n' % add)); count += 1
            add = ''
            if count%100 == 0:
                file.write('sleep 5m\n')
else:
    file.write(str('bsub -R "rusage[mem=4096]" -W 24:00 "%s"\n' % add))

file.close()

