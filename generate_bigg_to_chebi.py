# -*- coding: utf-8 -*-

import pandas as pd
import re

content = pd.read_table("bigg_models_metabolites.txt")
bigg_id = []
chebi = []
for i,met in enumerate(content.database_links):
    if type(met) is str:
        if ('CHEBI' in met):
            bigg_id.append(content.bigg_id[i])
            add = re.search('(?<=CHEBI:)\d+', met).group(0)
            chebi.append(add)

mapping = pd.DataFrame({'bigg_id':bigg_id,'chebi':chebi})
mapping.to_csv('bigg_to_chebi2.csv')