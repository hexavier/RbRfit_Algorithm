# -*- coding: utf-8 -*-

import pandas as pd
import cobra
import alloregfit as arf
from time import time

cond = ['glc-NCM3722_1','glc-NQ1243_3','glc-NQ1243_4','glc-NQ1243_5','glu-NCM3722_9','glu-NQ393_11','glu-NQ393_12','glu-NQ393_13','glu-NQ393_14']
data_dir = 'Y:/users/Xavier/3_Karl_data/'
model = cobra.io.load_matlab_model(data_dir+"iJO1366.mat")
fluxes = pd.read_excel(data_dir+"fluxes_GS_nozeros.xlsx",index_col="name")[cond]
metabolites = pd.read_excel(data_dir+"merged_metabolites.xlsx",index_col="name")[cond]
metabolites_sd = pd.read_excel(data_dir+"merged_metabolites_sd.xlsx",index_col="name")[cond]
proteins = pd.read_excel(data_dir+"proteome_noNaN.xlsx",index_col="name")[cond]
mapping = pd.read_table(data_dir+"ECOLI_83333_idmapping.dat",header=None)
rxn_id = open(data_dir+'reactions_GS.txt').read().splitlines()
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")

s = time()
#binding_site = arf.get_binding_sites(rxn_id,model)
#binding_site = [[['dhap_c', 'fdp_c', 'g3p_c']], [['fum_c', 'mal__L_c']], [['6pgl_c', 'g6p_c'], ['nadp_c', 'nadph_c']], [['6pgc_c', 'ru5p__D_c'], ['nadp_c', 'nadph_c'], ['co2_c']], [['mal__L_c', 'oaa_c'], ['nad_c', 'nadh_c']], [['nad_c', 'nadh_c'], ['nadp_c', 'nadph_c']], [['adp_c', 'atp_c'], ['f6p_c', 'fdp_c']], [['f6p_c', 'g6p_c']], [['adp_c', 'atp_c'], ['pep_c', 'pyr_c']], [['g3p_c', 'xu5p__D_c'], ['r5p_c', 's7p_c']], [['dhap_c', 'g3p_c']]]
summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites,metab_sd=metabolites_sd)
candidates = arf.define_candidates(rxn_id,reg_coli,metabolites,metabolites_sd)
markov_par = {'freq':2,'nrecord':2,'burn_in':0} # Record once every 20 samples, 200 samples, skip 0 first samples
e = time()
print('Load summary and candidates: %fs' % (e-s))

#%% Run fit with regulators
s = time()
results = arf.fit_reactions(summary,model,markov_par,candidates)#,candidates,maxreg=1,coop=True)
e = time()
print('Fit reaction MCMC-NNLS: %fs' % (e-s))

#results.to_pickle(data_dir+'GS_fit_1_candidates.pickle')