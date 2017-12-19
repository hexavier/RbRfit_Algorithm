# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import cobra
from numpy.random import seed
import numpy as np
import alloregfit as arf

data_dir = "//imsbnas.ethz.ch/Sauer1/users/Xavier/2_new_simmer_test/"
model = cobra.io.sbml.create_cobra_model_from_sbml_file(data_dir+"ecoli_core_model.xml")
fluxes = pd.read_excel(data_dir+"fluxes.xlsx",index_col="name")
metabolites = pd.read_excel(data_dir+"metabolites.xlsx",index_col="name")
proteins = pd.read_excel(data_dir+"proteins.xlsx",index_col="name")
rxn_id = open(data_dir+'reactions.txt').read().splitlines()
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")
binding_site = [[['fum_c', 'mal_L_c']], [['6pgc_c', 'ru5p_D_c'], ['nadp_c', 'nadph_c'], ['co2_c']], [['mal_L_c', 'oaa_c'], ['nad_c', 'nadh_c']], [['atp_c', 'adp_c'], ['f6p_c', 'fdp_c']], [['g6p_c', 'f6p_c']], [['atp_c', 'amp_c'], ['pyr_c', 'pep_c']], [['adp_c', 'atp_c'], ['pep_c', 'pyr_c']], [['r5p_c', 'ru5p_D_c']]]

summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites,binding_site=binding_site)
candidates = arf.define_candidates(rxn_id,reg_coli,metabolites)
markov_par = {'freq':10,'nrecord':20,'burn_in':0}

global random
random = seed(1)
results = arf.fit_reactions(summary,model,markov_par,candidates,sampleNaN=False,maxreg=1,coop=True)

class TestAlloRegFit(unittest.TestCase):

    def test_size(self):
        self.assertEqual(summary.shape, (8, 10))
        self.assertEqual(candidates.shape, (8, 8))
        self.assertEqual(results.shape, (38, 11))
        
    def test_results(self):
        mean = np.mean(results['best_lik'])
        self.assertEqual(mean, -16.595871951401392)
        self.assertTrue(results['idx'].iloc[0]==4)
        self.assertTrue(results['idx'].iloc[-1]==7)

if __name__ == '__main__':
    unittest.main()