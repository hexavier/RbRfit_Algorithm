# -*- coding: utf-8 -*-

import pandas as pd
import cobra
import alloregfit as arf
from time import time

data_dir = "C:/Users/user/polybox/MASTER/THESIS/2_new_simmer_test/"
model = cobra.io.sbml.create_cobra_model_from_sbml_file(data_dir+"ecoli_core_model.xml")
#model = pd.read_excel(data_dir+"ecoli_core_model.xls",index_col="METABOLITE")
fluxes = pd.read_excel(data_dir+"fluxes.xlsx",index_col="name")
#fluxes_sd = pd.read_excel(data_dir+"fluxes.xlsx",sheetname=1,index_col="name")
metabolites = pd.read_excel(data_dir+"metabolites.xlsx",index_col="name")
#metabolites_sd = pd.read_excel(data_dir+"metabolites.xlsx",sheetname=1,index_col="name")
proteins = pd.read_excel(data_dir+"proteins.xlsx",index_col="name")
mapping = pd.read_table(data_dir+"ECOLI_83333_idmapping.dat",header=None)
rxn_id = open(data_dir+'reactions.txt').read().splitlines()
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")

s = time()
#binding_site = arf.get_binding_sites(rxn_id,model)
binding_site = [[['fum_c', 'mal_L_c']], [['6pgc_c', 'ru5p_D_c'], ['nadp_c', 'nadph_c'], ['co2_c']], [['mal_L_c', 'oaa_c'], ['nad_c', 'nadh_c']], [['atp_c', 'adp_c'], ['f6p_c', 'fdp_c']], [['g6p_c', 'f6p_c']], [['atp_c', 'amp_c'], ['pyr_c', 'pep_c']], [['adp_c', 'atp_c'], ['pep_c', 'pyr_c']], [['r5p_c', 'ru5p_D_c']]]
summary,bools = arf.define_reactions(rxn_id,model,binding_site,fluxes,proteins,metabolites)
candidates = arf.define_candidates(rxn_id,reg_coli,metabolites,bools)
markov_par = {'freq':20,'nrecord':20,'burn_in':0} # Record once every 20 samples, 200 samples, skip 0 first samples
e = time()
print('Load summary and candidates: %fs' % (e-s))

#%% Run fit with regulators
s = time()
results = arf.fit_reactions(summary,model,markov_par)#,candidates,maxreg=1,coop=True)
e = time()
print('Fit reaction MCMC-NNLS: %fs' % (e-s))
