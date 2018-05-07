# -*- coding: utf-8 -*-

import alloregfit as arf
import pandas as pd
import cobra

results = pd.read_pickle("results_all.pickle")
gold = pd.read_csv("gold_standard.csv",index_col="reaction")
mapping = pd.read_table("ECOLI_83333_idmapping.dat",header=None)
cond = ['glc-NCM3722_1','glc-NQ1243_3','glc-NQ1243_4','glc-NQ1243_5','glu-NCM3722_9','glu-NQ393_11','glu-NQ393_12','glu-NQ393_13','glu-NQ393_14']
data_dir = '/cluster/home/hexavier/alloregfit/4_Karl_data/'
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")
model = cobra.io.load_matlab_model(data_dir+"iJO1366.mat")
fluxes = pd.read_excel(data_dir+"fluxes_FVA.xlsx",sheet_name=['min','max'],index_col="name")
fluxes['min'] = fluxes['min'][cond]; fluxes['max'] = fluxes['max'][cond]
metabolites = pd.read_excel(data_dir+"merged_metabolites.xlsx",index_col="name")[cond]
metabolites_sd = pd.read_excel(data_dir+"merged_metabolites_sd.xlsx",index_col="name")[cond]
proteins = pd.read_excel(data_dir+"proteome_noNaN.xlsx",index_col="name")[cond]
rxn_id = open(data_dir+'reactions_bindsites.txt').read().splitlines()
summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites,metab_sd=metabolites_sd)

val_bycond,reduced = arf.validate_bycond(results,summary,candidates=metabolites)
val_bycond.to_pickle('validation_by_condition.pickle')
reduced.to_pickle('validated_results.pickle')

shortlist= arf.rankresults(reduced,model)
shortlist.to_pickle(shortlist.pickle')