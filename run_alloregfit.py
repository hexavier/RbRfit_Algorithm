# -*- coding: utf-8 -*-

import pandas as pd
import cobra
import alloregfit as arf

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

summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites)
markov_par = {'freq':20,'nrecord':200,'burn_in':0} # Record once every 20 samples, 200 samples, skip 0 first samples

results = arf.fit_reactions(summary,model,markov_par)
