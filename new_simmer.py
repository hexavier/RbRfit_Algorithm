# -*- coding: utf-8 -*-

#%% Import modules
import pandas as pd

#%% Define reactions
# For each of the reactions, the function creates a data frame where every row constitutes a reaction.

# Function definition

def define_reactions(rxn_id, model, fluxes, prot, metab):
    mapping = pd.read_table("ECOLI_83333_idmapping.dat",header=None)
    reaction = []
    rxn_name = []
    reactant = []
    product = []
    enzyme = []
    flux = []
    for i in range(len(rxn_id)):
        # Name value
        rxn_name.append(model.reactions.get_by_id(rxn_id[i]).name)
        # Reaction value
        reaction.append(model.reactions.get_by_id(rxn_id[i]).reaction)
        # Reactant values
        react = model.reactions.get_by_id(rxn_id[i]).reactants
        react_df = []
        name = []
        for j in range(len(react)):
            if (any(react[j].id[:-2] in s for s in ['h','h2o'])==0):
                react_df.append(metab.loc[react[j].id[:-2]].values)
                name.append(react[j].id[:-2])
        react_df = pd.DataFrame(react_df,columns = metab.columns, index = name)
        reactant.append(react_df)
        # Product values
        prod = model.reactions.get_by_id(rxn_id[i]).products
        prod_df = []
        name = []
        for j in range(len(prod)):
            if (any(prod[j].id[:-2] in s for s in [metab.index.values])&(any(prod[j].id[:-2] in s for s in ['h','h2o'])==0)):
                prod_df.append(metab.loc[prod[j].id[:-2]].values)
                name.append(prod[j].id[:-2])
        prod_df = pd.DataFrame(prod_df,columns = metab.columns, index = name)
        product.append(prod_df)
        # Enzyme values
        enz = list(model.reactions.get_by_id(rxn_id[i]).genes)
        enz_df = []
        name = []
        for j in range(len(enz)):
            gene = mapping[mapping[2]==enz[j].id][0].reset_index()
            gene = list(mapping[(mapping[0]==gene[0][0]) & (mapping[1]=='Gene_Name')][2])
            if any(gene[0] in s for s in [prot.index.values]):
                enz_df.append(prot.loc[gene[0]].values)
                name.append(gene)
        enz_df = pd.DataFrame(enz_df, columns = prot.columns, index = name)
        enzyme.append(enz_df)
        # Flux values
        flux.append(pd.DataFrame([fluxes.loc[rxn_id[i]].values],columns = fluxes.columns, index = [rxn_id[i]]))
    
    summary = pd.DataFrame({'idx':range(len(rxn_id)),'reaction':reaction,'rxn_id':rxn_id,\
                            'name':rxn_name,'reactant':reactant,'product':product,\
                            'enzyme':enzyme,'flux':flux})
    summary.set_index('idx')
    return summary
