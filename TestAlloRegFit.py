# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import cobra
from numpy.random import seed
import numpy as np
import alloregfit as arf

data_dir = "C:/Users/user/polybox/MASTER/THESIS/2_new_simmer_test/"
model = cobra.io.sbml.create_cobra_model_from_sbml_file(data_dir+"ecoli_core_model.xml")
fluxes = pd.read_excel(data_dir+"fluxes.xlsx",index_col="name")
metabolites = pd.read_excel(data_dir+"metabolites.xlsx",index_col="name")
proteins = pd.read_excel(data_dir+"proteins.xlsx",index_col="name")
rxn_id = open(data_dir+'reactions.txt').read().splitlines()
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")

summary,bools = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites)
candidates = arf.define_candidates(rxn_id,reg_coli,metabolites,bools)
markov_par = {'freq':10,'nrecord':20,'burn_in':0}

global random
random = seed(1)
results = arf.fit_reactions(summary,model,markov_par,candidates,maxreg=1,coop=True)

class TestAlloRegFit(unittest.TestCase):

    def test_size(self):
        self.assertEqual(summary.shape, (8, 7))
        self.assertEqual(candidates.shape, (8, 4))
        self.assertEqual(results.shape, (22, 5))
        
    def test_results(self):
        mean = np.mean(results['best_lik'])
        self.assertEqual(mean, -20.469714179933025)
        self.assertTrue(results.index.values[0]==4.0)
        self.assertTrue(results.index.values[-1]==0.0)

if __name__ == '__main__':
    unittest.main()