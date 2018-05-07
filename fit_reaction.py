# -*- coding: utf-8 -*-

import sys
import pandas as pd
import cobra
import alloregfit as arf
from os import listdir
from os.path import isfile, join

idx = sys.argv[1]
i = sys.argv[2]
model_path = sys.argv[3]

def fit_reaction(idx,i,summary,model,markov_par,candidates=None,candidates_sd=None,priorKeq=False,maxreg=1,coop=False,sampleNaN=True):
    onlyfiles = [f for f in listdir(str('results%i'%idx)) if isfile(join(str('results%i'%idx), f))]
    if 'reaction_forms.pickle' not in onlyfiles:
        expr,parameters,regulator = arf.write_rate_equations(idx,summary,model,candidates,maxreg,coop)
        file = pd.DataFrame({'expr':expr,'parameters':parameters,'regulator':regulator})
        file.to_pickle(str('results%s/reaction_forms.pickle' % idx))
    else:
        file = pd.read_pickle(str('results%s/reaction_forms.pickle' % idx))
        expr = list(file['expr']); parameters = list(file['parameters']); regulator = list(file['regulator'])

    parameters[i] = arf.build_priors(parameters[i],idx,summary,model,priorKeq,candidates,sampleNaN)
    track,bool_all = arf.fit_reaction_MCMC(idx,markov_par,parameters[i],summary,expr[i],candidates,regulator[i],sampleNaN)
    if track is None:
        results = pd.DataFrame(columns=['idx','reaction','rxn_id','regulator','equation',\
                                    'meas_flux','pred_flux','best_fit','best_lik','lik_cond'])
    else:
        max_lik = max(track['likelihood'].values)
        max_par = track[track['likelihood'].values==max_lik]
        par_bool = [',' not in s for s in max_par.columns]; par_bool[-3:]=[False]*3
        uncertainty = arf.cal_uncertainty(idx, expr[i], max_par.loc[:,par_bool],summary,candidates,regulator[i],candidates_sd)
        add = {'idx':idx,'reaction':summary['reaction'][idx],'rxn_id':summary['rxn_id'][idx],\
               'regulator':regulator[i],'equation':(expr[i]['vmax']*expr[i]['occu']),'meas_flux':summary['flux'][idx].loc[:,bool_all],\
               'pred_flux':max_par.iloc[:,-2].values,'uncertainty':uncertainty[0],'best_fit':max_par.iloc[:,:-3],'best_lik':max_lik,\
               'lik_cond':max_par.iloc[:,-1].values[0]}
        results = pd.DataFrame([add])
    return results
                
model = cobra.io.load_matlab_model(model_path)
summary = pd.read_pickle('summary.pickle')
candidates = pd.read_pickle('candidates.pickle')
cand_sd = pd.read_pickle('candidates_sd.pickle')
markov_par = {'freq':20,'nrecord':200,'burn_in':0}

results = fit_reaction(int(idx),int(i),summary,model,markov_par,candidates,candidates_sd=cand_sd)

results.to_pickle(str('results%s/reg_%s.pickle' % (idx,i)))