# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:06:20 2017

@author: user
"""

import pandas as pd
import cobra
import new_simmer as sim

data_dir = "C:/Users/user/polybox/MASTER/THESIS/2_new_simmer_test/"
model = cobra.io.sbml.create_cobra_model_from_sbml_file(data_dir+"ecoli_core_model.xml")
#model = pd.read_excel(data_dir+"ecoli_core_model.xls",index_col="METABOLITE")
fluxes = pd.read_excel(data_dir+"fluxes.xlsx",index_col="name")
metabolites = pd.read_excel(data_dir+"metabolites.xlsx",index_col="name")
proteins = pd.read_excel(data_dir+"proteins.xlsx",index_col="name")
mapping = pd.read_table(data_dir+"ECOLI_83333_idmapping.dat",header=None)
rxn_id = open(data_dir+'reactions.txt').read().splitlines()

summary = sim.define_reactions(rxn_id,model,fluxes,proteins,metabolites)