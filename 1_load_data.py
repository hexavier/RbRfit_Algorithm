# -*- coding: utf-8 -*-

import pandas as pd
import cobra
import alloregfit as arf

cond = ['glc-NCM3722_1','glc-NQ1243_3','glc-NQ1243_4','glc-NQ1243_5','glu-NCM3722_9','glu-NQ393_11','glu-NQ393_12','glu-NQ393_13','glu-NQ393_14']
data_dir = "/cluster/home/hexavier/alloregfit/4_Karl_data/"
model = cobra.io.load_matlab_model(data_dir+"iJO1366.mat")
fluxes = pd.read_excel(data_dir+"fluxes_FVA.xlsx",sheet_name=['min','max'],index_col="name")
fluxes['min'] = fluxes['min'][cond]; fluxes['max'] = fluxes['max'][cond]
metabolites = pd.read_excel(data_dir+"merged_metabolites.xlsx",index_col="name")[cond]
metabolites_sd = pd.read_excel(data_dir+"merged_metabolites_sd.xlsx",index_col="name")[cond]
proteins = pd.read_excel(data_dir+"proteome_noNaN.xlsx",index_col="name")[cond]
rxn_id = open(data_dir+'reactions_bindsites.txt').read().splitlines()
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")

summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites,metab_sd=metabolites_sd)

summary.to_pickle('summary.pickle')
metabolites.to_pickle('candidates.pickle')
metabolites_sd.to_pickle('candidates_sd.pickle')
