# -*- coding: utf-8 -*-

#%% Import modules
import pandas as pd
import sympy as sym
import numpy as np
from obonet import read_obo
from numpy.random import uniform,normal
from scipy.stats import norm,chi2, pearsonr
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from seaborn import heatmap
from matplotlib import cm
import re
from sklearn.linear_model import LogisticRegression

#%% Extract available molecules
# For all the molecules involved in the reaction, extract the corresponding omics data.
# Inputs: molecules to extract info, omics dataframe, and type of molecule (reactant,
# product or enzyme).

def extract_info_df(molecules, dataset, sd, mol_type):
    mol_df, sd_df, names = ([] for l in range(3))
    if mol_type=='enzyme':
        mapping = pd.read_table("ECOLI_83333_idmapping.dat",header=None)
        for j in range(len(molecules)):
            gene = mapping[mapping[2]==molecules[j].id][0].reset_index()
            gene = list(mapping[((mapping[0]==gene[0][0]) & (mapping[1]=='Gene_Name'))][2]) # obtain gene names from b numbers
            if any(gene[0] in s for s in [dataset.index.values]): # extract abundance info if gene is in the proteomic dataset
                mol_df.append(dataset.loc[gene[0]].values)
                if (sd is not None): sd_df.append(sd.loc[gene[0]].values) # extract also standard deviation
                names.append(gene[0])
    else:
        for j in range(len(molecules)):
            met = molecules[j].id[:-2] #strip compartment letter from id
            if ((mol_type=='reactant')and(any(met in s for s in ['h','h2o'])==0)) or \
            ((mol_type=='product')and(any(met in s for s in [dataset.index.values]) and \
              (any(met in s for s in ['h','h2o'])==0))): # extract abundance data of substrates and products when present
                    mol_df.append(dataset.loc[met].values)
                    if (sd is not None): sd_df.append(sd.loc[met].values) # extract also standard deviation
                    names.append(molecules[j].id)
    mol_df = pd.DataFrame(mol_df,columns = dataset.columns, index = names) # include retrieved data in a pandas daaframe, as well as standard deviations
    if sd_df:
        sd_df = pd.DataFrame(sd_df,columns = dataset.columns, index = names)
    else:
        sd_df = pd.DataFrame(np.zeros(mol_df.shape),columns = dataset.columns, index = names)
    return mol_df,sd_df
    
#%% Determine binding site
# For all metabolites involved in reaction, determine the binding site.
# Inputs: reactants, products, model.
    
def get_binding_sites(rxn_id,model):
    obo_map = read_obo("chebi_lite.obo")
    bigg_chebi = pd.read_csv("bigg_to_chebi2.csv", index_col = 'bigg_id')
    binding_site = []
    for i in range(len(rxn_id)):
        all_met = [x.id for x in model.reactions.get_by_id(rxn_id[i]).metabolites.keys()]
        all_met = [s for s in all_met if s not in ['h_c','h2o_c','pi_c']] # obtain a list of metabolites involved in the reaction.
        bs_array = np.zeros([len(all_met),len(all_met)]) # initialize boolean array with binding site info
        all_chebi = list(map(lambda x: int(bigg_chebi.loc[x]),all_met)) # list of chebi identifiers of the respective metabolites
        id_to_altid = {id_:data['alt_id'] for id_, data in obo_map.nodes(data=True) if any('alt_id' in s for s in list(data.keys()))} # create dictionary that maps chebi IDs to alternative chebi IDs
        altid_to_id = {}
        for k,v in id_to_altid.items(): # invert id_to_altid dictionary, mapping alternative IDs to IDs
            for alt in v:
                altid_to_id[alt]=k
        ori_id = [altid_to_id[str('CHEBI:%i' % ori)] if any(str('CHEBI:%i' % ori) in s for s in list(altid_to_id.keys())) else str('CHEBI:%i'%ori) for ori in all_chebi] # from the list of chebi IDs of our metabolites, retrieve the original ID if it is an alternative chebi ID
        for r,met1 in enumerate(ori_id): # loop through all possible pairwise combinations of metabolites and look for structural similarities based on the chebi ID
            neigh1 = obo_map.successors(met1)
            for j,met2 in enumerate(ori_id):
                neigh2 = obo_map.successors(met2)
                if any([True for n1, n2 in zip(neigh1, neigh2) if n1 == n2]) and r<=j:
                    bs_array[r,j] = 1 # if yes, then update the binding site boolean array
        bind_site = []
        all_met = np.array(all_met)
        while (all_met.shape[0])>1:
            bind_site.append(list(all_met[bs_array[0]==1])) # put all metabolites sharing structure with the first metabolite in the same binding site
            all_met = all_met[bs_array[0,:]==0] # keep only metabolites that were not sharing structure for the next iterations
            bs_array = bs_array[1:,bs_array[0,:]==0] # delete first row and structurally similar metabolites from the binding site boolean array
        else:
            if (all_met.shape[0])>0:
                bind_site.append([str(all_met[0])]) # if last metbolite in the list is alone, add it in a separate binding site
            
        print(model.reactions.get_by_id(rxn_id[i]).reaction)
        print(bind_site)
        
        # The automatic binding site inference is error prone. Check the result and, if wrong, set bonding site manually.
        yes_no = input('Is this binding site correct?[y/n] --> ')
        if yes_no=='n':
            all_met = [x.id for x in model.reactions.get_by_id(rxn_id[i]).metabolites.keys()]
            all_met = [s for s in all_met if s not in ['h_c','h2o_c','pi_c']]
            num_bs = input('How many binding sites are there? --> ')
            bind_site = []
            for r,met in enumerate(all_met):
                print('%i: %s' % (r+1,met))
            for j in range(int(num_bs)):
                if j==0:
                    bs = input('Molecules in the first binding site. Ex:1,3 --> ')
                    add = []
                    for r in bs.split(','):
                        add.append(str('%s' % all_met[int(r)-1]))
                    bind_site.append(add)
                elif j==1:
                    bs = input('Molecules in the second binding site. Ex:1,3 --> ')
                    add = []
                    for r in bs.split(','):
                        add.append(str('%s' % all_met[int(r)-1]))
                    bind_site.append(add)
                elif j==2:
                    bs = input('Molecules in the third binding site. Ex:1,3 --> ')
                    add = []
                    for r in bs.split(','):
                        add.append(str('%s' % all_met[int(r)-1]))
                    bind_site.append(add)
                elif j>2:
                    bs = input(str('Molecules in the %ith binding site. Ex:1,3 --> ' % (j+1)))
                    add = []
                    for r in bs.split(','):
                        add.append(str('%s' % all_met[int(r)-1]))
                    bind_site.append(add)
        binding_site.append(bind_site)
                
    return binding_site

#%% Define reactions
# For each of the reactions, the function creates a data frame where every row constitutes a reaction.
# Inputs: list of reaction ids that will be analyzed, stoichiometric model, DataFrame containing fluxes x conditions,
# DataFrame containing prot x cond, and DataFrame with metabolites x cond.

def define_reactions(rxn_id, model, fluxes, prot, metab, prot_sd=None, metab_sd=None,binding_site=None):
    reaction, reactant, reactant_sd, product, product_sd, enzyme, enzyme_sd, flux, bs = ([] for l in range(9))
    
    for i in range(len(rxn_id)):
        # Reaction value
        reaction.append(model.reactions.get_by_id(rxn_id[i]).reaction)
        # Reactant values
        react = model.reactions.get_by_id(rxn_id[i]).reactants
        react_df,react_sd = extract_info_df(react,metab,metab_sd,'reactant')
        # Product values
        prod = model.reactions.get_by_id(rxn_id[i]).products
        prod_df,prod_sd = extract_info_df(prod,metab,metab_sd,'product')
        # Enzyme values
        enz = list(model.reactions.get_by_id(rxn_id[i]).genes)
        enz_df,enz_sd = extract_info_df(enz,prot,prot_sd,'enzyme')
        # Append all data
        reactant.append(react_df.copy())
        reactant_sd.append(react_sd.copy())
        product.append(prod_df.copy())
        product_sd.append(prod_sd.copy())
        enzyme.append(enz_df.copy())
        enzyme_sd.append(enz_sd.copy())
        # Fluxes. Append either as min/max values from FVA or as point estimates from FBA
        if isinstance(fluxes,pd.DataFrame):
            flux.append(pd.DataFrame([fluxes.loc[rxn_id[i]].values].copy(),columns = fluxes.columns, index = [rxn_id[i]]))
        elif isinstance(fluxes,dict):
            flux_df = pd.DataFrame([fluxes['min'].loc[rxn_id[i]].values,fluxes['max'].loc[rxn_id[i]].values],columns = fluxes['min'].columns, index = ['min','max'])
            flux.append(flux_df)
        # If no binding site information is included, consider one unique binding site
        if binding_site is None:
            bs.append([list(react_df.index.values)+list(prod_df.index.values)])
    
    if binding_site is None:
        binding_site=bs
        
    summary = pd.DataFrame({'idx':range(len(rxn_id)),'reaction':reaction,'rxn_id':rxn_id,\
                            'reactant':reactant,'reactant_sd':reactant_sd,'product':product,\
                            'product_sd':product_sd,'enzyme':enzyme,'enzyme_sd':enzyme_sd,\
                            'flux':flux,'binding_site':binding_site})
    summary = summary.set_index('idx')
    return summary

#%% Define candidates
# For each reaction, a table with the regulators is created. 
# Inputs: list of reaction ids that will be analyzed, DataFrame with regulators
# for all rxn_id in E. coli, DataFrame with metabolites x cond, and DataFrame with
# regulators for all rxn_id in other organisms (optional).
    
def define_candidates(rxn_id,reg_coli,metab,metab_sd=None,reg_other=None):
    act_coli,act_coli_sd, inh_coli,inh_coli_sd, act_other,act_other_sd, inh_other,inh_other_sd = ([] for l in range(8))
    for i in range(len(rxn_id)):
        if (any(rxn_id[i].lower() in s for s in [reg_coli.index.values])):
            cand_coli = reg_coli.loc[[rxn_id[i].lower()]].reset_index() # look for candidates involving the reaction
            act_coli_df,act_coli_sd_df,name_act_coli,inh_coli_df,inh_coli_sd_df,name_inh_coli = ([] for l in range(6))
            for j,met in enumerate(list(cand_coli['metab'])):
                if (any(met in s for s in [metab.index.values])): # add candidates only if they are present in metabolomic dataset
                    if cand_coli['mode'][j] == '-':
                        inh_coli_df.append(metab.loc[met].values) # add abundancies
                        if (metab_sd is not None):
                            inh_coli_sd_df.append(metab_sd.loc[met].values) # add standard deviation
                        name_inh_coli.append(met+'_c') # add bigg metabolite name
                    elif cand_coli['mode'][j] == '+':
                        act_coli_df.append(metab.loc[met].values) # add abundancies
                        if (metab_sd is not None):
                            act_coli_sd_df.append(metab_sd.loc[met].values) # add standard deviation
                        name_act_coli.append(met+'_c') # add bigg metabolite name
            
            # Add all retrieved info in structure
            inh_coli_df = pd.DataFrame(inh_coli_df,columns = metab.columns, index=name_inh_coli)
            act_coli_df = pd.DataFrame(act_coli_df,columns = metab.columns, index=name_act_coli)
            if metab_sd is None:
                inh_coli_sd_df = np.zeros(inh_coli_df.shape)
                act_coli_sd_df = np.zeros(act_coli_df.shape)
            inh_coli_sd_df = pd.DataFrame(inh_coli_sd_df,columns = metab.columns, index=name_inh_coli)
            act_coli_sd_df = pd.DataFrame(act_coli_sd_df,columns = metab.columns, index=name_act_coli)
            if act_coli_df.empty:
                act_coli.append('No data available for the candidate activators.')
                act_coli_sd.append('No data available for the candidate activators.')
            else:
                act_coli_df = act_coli_df.reset_index().drop_duplicates().set_index('index'); act_coli.append(act_coli_df.copy())
                act_coli_sd_df = act_coli_sd_df.reset_index().drop_duplicates().set_index('index'); act_coli_sd.append(act_coli_sd_df.copy())
            if inh_coli_df.empty:
                inh_coli.append('No data available for the candidate activators.')
                inh_coli_sd.append('No data available for the candidate activators.')
            else:
                inh_coli_df = inh_coli_df.reset_index().drop_duplicates().set_index('index'); inh_coli.append(inh_coli_df.copy())
                inh_coli_sd_df = inh_coli_sd_df.reset_index().drop_duplicates().set_index('index'); inh_coli_sd.append(inh_coli_sd_df.copy())
        else:
            act_coli.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
            act_coli_sd.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
            inh_coli.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
            inh_coli_sd.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
        if reg_other is None:
            act_other.append([None]*len(rxn_id))
            act_other_sd.append([None]*len(rxn_id))
            inh_other.append([None]*len(rxn_id))
            inh_other_sd.append([None]*len(rxn_id))
        else: # do the same for regulators in other organisms
            if (any(rxn_id[i].lower() in s for s in [reg_other.index.values])):
                cand_other = reg_other.loc[[rxn_id[i].lower()]].reset_index()
                act_other_df,act_other_sd_df,name_act_other,inh_other_df,inh_other_sd_df,name_inh_other = ([] for l in range(6))
                for j,met in enumerate(list(cand_other['metab'])):
                    if (any(met in s for s in [metab.index.values])):
                        if cand_other['mode'][j] == '-':
                            inh_other_df.append(metab.loc[met].values)
                            if (metab_sd is not None):
                                inh_other_sd_df.append(metab_sd.loc[met].values)
                            name_inh_other.append(met+'_c')
                        elif cand_other['mode'][j] == '+':
                            act_other_df.append(metab.loc[met].values)
                            if (metab_sd is not None):
                                act_other_sd_df.append(metab_sd.loc[met].values)
                            name_act_other.append(met+'_c')
                inh_other_df = pd.DataFrame(inh_other_df,columns = metab.columns, index=name_inh_other)
                act_other_df = pd.DataFrame(act_other_df,columns = metab.columns, index=name_act_other)
                if metab_sd is None:
                    inh_other_sd_df = np.zeros(inh_other_df.shape)
                    act_other_sd_df = np.zeros(act_other_df.shape)
                inh_other_sd_df = pd.DataFrame(inh_other_sd_df,columns = metab.columns, index=name_inh_other)
                act_other_sd_df = pd.DataFrame(act_other_sd_df,columns = metab.columns, index=name_act_other)
                if act_other_df.empty:
                    act_other.append('No data available for the candidate activators.')
                    act_other_sd.append('No data available for the candidate activators.')
                else:
                    act_other_df = act_other_df.reset_index().drop_duplicates().set_index('index'); act_other.append(act_other_df.copy())
                    act_other_sd_df = act_other_sd_df.reset_index().drop_duplicates().set_index('index'); act_other_sd.append(act_other_sd_df.copy())
                if inh_other_df.empty:
                    inh_other.append('No data available for the candidate activators.')
                    inh_other_sd.append('No data available for the candidate activators.')
                else:
                    inh_other_df = inh_other_df.reset_index().drop_duplicates().set_index('index'); inh_other.append(inh_other_df.copy())
                    inh_other_sd_df = inh_other_sd_df.reset_index().drop_duplicates().set_index('index'); inh_other_sd.append(inh_other_sd_df.copy())
            else:
                act_other.append('No candidate regulators for %s.' % rxn_id[i])
                act_other_sd.append('No candidate regulators for %s.' % rxn_id[i])
                inh_other.append('No candidate regulators for %s.' % rxn_id[i])
                inh_other_sd.append('No candidate regulators for %s.' % rxn_id[i])
    candidates = pd.DataFrame({'idx':range(len(rxn_id)),'act_coli':act_coli,'act_coli_sd':act_coli_sd,\
                               'inh_coli':inh_coli,'inh_coli_sd':inh_coli_sd,'act_other':act_other,\
                               'act_other_sd':act_other_sd,'inh_other':inh_other,'inh_other_sd':inh_other_sd})
    candidates = candidates.set_index('idx')
    return candidates        

#%% Write regulator expression
# For each regulator, write the regulatory expression to add.
# Inputs: list of regulators and their +/- effect.
    
def write_reg_expr(regulators,reg_type,coop=False):
    add, newframe, reglist = ([] for l in range(3))
    for reg in regulators:
        R = str('c_%s' % reg) # symbol for the regulator
        K = str('K_reg_%s' % reg) # symbol for the regulatory constant
        if coop is False:
            if reg_type=='activator':
                add.append(sym.sympify(R+'/('+R+'+'+K+')')) # create symbolic expression
                reglist.append('ACT:'+reg) # add regulator name
            elif reg_type=='inhibitor':
                add.append(sym.sympify('1/(1+('+R+'/'+K+'))')) # create symbolic expression
                reglist.append('INH:'+reg) # add regulator name
            new_par = [K]; new_spe = [reg]; new_spetype = ['met']
        elif coop is True:
            n = str('n_%s' % reg) # symbol for the Hill coefficient
            if reg_type=='activator':
                add.append(sym.sympify(R+'**'+n+'/('+R+'**'+n+'+'+K+'**'+n+')')) # create symbolic expression
                reglist.append('ACT:'+reg) # add regulator name
            elif reg_type=='inhibitor':
                add.append(sym.sympify('1/(1+('+R+'/'+K+')**'+n+')')) # create symbolic expression
                reglist.append('INH:'+reg) # add regulator name
            new_par = [K,n]; new_spe = [reg,'hill']; new_spetype = ['met','hill']
        newframe.append(pd.DataFrame({'parameters':new_par,'species':new_spe,'speciestype':new_spetype}))
    return add, newframe, reglist

#%% Add regulators
# For all kind of regulators, generate a structure containing all expressions to add, and their respective parameters.
# Inputs: candidates dataframe
    
def add_regulators(idx,candidates,coop=False):
    add, newframe, reg = ([] for l in range(3))
    # If only selected candidates are included in the analysis, "candidates" is the candidates dataframe.
    if (list(candidates.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
        'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']):
        if isinstance(candidates['act_coli'][idx],pd.DataFrame):
            act_coli = list(candidates['act_coli'][idx].index) # list of activatprs in E. coli
            add1, newframe1, reg1 = write_reg_expr(act_coli,'activator',coop) 
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['inh_coli'][idx],pd.DataFrame):
            inh_coli = list(candidates['inh_coli'][idx].index) # list of inhibitors in E. coli
            add1, newframe1, reg1 = write_reg_expr(inh_coli,'inhibitor',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['act_other'][idx],pd.DataFrame):
            act_other = list(candidates['act_other'][idx].index) # list of activators
            add1, newframe1, reg1 = write_reg_expr(act_other,'activator',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['inh_other'][idx],pd.DataFrame):
            inh_other = list(candidates['inh_other'][idx].index) # list of inhibitors
            add1, newframe1, reg1 = write_reg_expr(inh_other,'inhibitor',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
    # If all metabolites are included systematically, the metabolomics dataframe is inputted directly.
    else:
        cand = list(map(lambda x: x+'_c',list(candidates.index))) # list of all regulators
        add1, newframe1, reg1 = write_reg_expr(cand,'activator',coop) # as activators
        add2, newframe2, reg2 = write_reg_expr(cand,'inhibitor',coop) # as inhibitors
        add.extend(add1+add2); newframe.extend(newframe1+newframe2); reg.extend(reg1+reg2)
    return add, newframe, reg
    
#%% Write Rate Equations
# For each of the models, write one rate equation expression. If products are available, include them.
# Inputs: summary generated by define_reactions, idx defining the reaction that is analyzed,
# stoichiometric model and candidates dataframe (optional).

def write_rate_equations(idx,summary, model, candidates=None, nreg=1, coop=False):
    parameters, species, speciestype = ([] for i in range(3))
    # Define Vmax expression:
    enzyme = list(summary['enzyme'][idx].index)
    vmax = sym.sympify('0')
    for enz in enzyme: # loop through all the isoforms
        K = str('K_cat_%s' % enz)
        E = str('c_%s' % enz)
        vmax += sym.sympify(K+'*'+E)
        
    # Define occupancy term. Start with the numerator:
    reaction = model.reactions.get_by_id(summary['rxn_id'][idx])
    substrate = list(summary['reactant'][idx].index)
    num1 = sym.sympify('1') # first part of the numerator, including the substrate dissociation constants
    num2 = sym.sympify('1') # second part of the numerator, including the substrate concentrations to the power of their stoichiometric coefficient
    for sub in substrate: # loop through all the substrates
        K = str('K_%s' % sub)
        num1 *= sym.sympify(K)
        S = str('c_%s' % sub)
        exp = abs(reaction.get_coefficient(sub))
        num2 *= sym.sympify(S+'**'+str(exp))
        parameters.append(K), species.append(sub), speciestype.append('met')
    num1 = 1/num1            
    
    product = list(summary['product'][idx].index)
    if product: # include products only if they have been measured
        num3 = sym.sympify('1') # third part of the numerator, including the product concentrations to the power of their stoichiometric coefficient
        for prod in product: # loop through all detected products
            P = str('c_%s' % prod)
            exp = abs(reaction.get_coefficient(prod))
            num3 *= sym.sympify(P+'**'+str(exp))
        K_eq = sym.symbols('K_eq')
        parameters.append('K_eq'), species.append('K_eq'), speciestype.append('K_eq')
        num3 = (1/K_eq)*num3
        num = num1*(num2-num3) # put the three numerator parts together (if products)
    else:
        num = num1*num2 # put the two numerator parts together (if no products)
    
    # Define the denominator:
    den = sym.sympify('1')
    for i,site in enumerate(summary['binding_site'][idx]): # loop through all the binding sites
        den_site = sym.sympify('1')
        for met in summary['binding_site'][idx][i]: # loop through all metabolites in the binding site
            if any(met in s for s in substrate+product):
                exp = int(abs(reaction.get_coefficient(met)))
                for j in range(1, (exp+1)):
                    R = str('c_%s' % met)
                    K = str('K_%s' % met)
                    den_site += sym.sympify('('+R+'/'+K+')**'+str(j))
                    parameters.append(K), species.append(met), speciestype.append('met')
        den *= den_site
        
    # Paste all the parts together:
    expr = [{'vmax':vmax,'occu':(num/den)}]
    
    # Generate list of parameters:
    parframe = [pd.DataFrame({'parameters':parameters,'species':species,'speciestype':speciestype})]
    parframe[0].drop_duplicates('parameters',inplace=True) # remove duplicates of parameters that appear twice in formula
    parframe[0].reset_index(drop=True,inplace=True)
    
    regulator = ['']
    if (candidates is not None) and (nreg>=1):
        add, newframe, reg = add_regulators(idx,candidates,coop)
        add=[add[n] for n,i in enumerate(reg) if i not in reg[:n]] # remove duplicates in add list
        newframe=[newframe[n] for n,i in enumerate(reg) if i not in reg[:n]] # remove duplicates in newframe list
        reg=[i for n,i in enumerate(reg) if i not in reg[:n]] # remove duplicates in reg list
        for i in range(len(add)): # loop through all candidates
            expr.append({'vmax':vmax,'occu':add[i]*(num/den)}) # add regulation in reaction symbolic expression
            addframe = parframe[0].append(newframe[i]) # add regulator parameters in parframe
            addframe.drop_duplicates('parameters',inplace=True)
            addframe.reset_index(drop=True,inplace=True)
            parframe.append(addframe)
            regulator.append([reg[i]])
            if nreg>=2:
                for j in range(len(add)): # loop through the pairwise candidates
                    if i>j: # [i,j] is the same as [j,i], avoid unnecessary combinations
                        expr.append({'vmax':vmax,'occu':add[j]*add[i]*(num/den)}) # add regulation in reaction symbolic expression
                        addframe = parframe[0].append(newframe[i]) # add regulator parameters in parframe
                        addframe = addframe.append(newframe[j])
                        addframe.drop_duplicates('parameters',inplace=True)
                        addframe.reset_index(drop=True,inplace=True)
                        parframe.append(addframe)
                        regulator.append([reg[i],reg[j]])
    
    return expr,parframe,regulator

#%% Fill NaN values
# Sample NaN from existing values and add them to parameter table.
# Inputs: add dataframe, summary dataframe, reaction index, type of molecule.

def fill_nan(add,idx,summary,molecule):
    for i,mol in enumerate(summary[molecule][idx].index):
        isnan = np.isnan(summary[molecule][idx].loc[mol]) # find nan in summary table
        if any(isnan):
            for j,cond in enumerate(summary[molecule][idx].columns[isnan]):
                par1 = np.nanmedian(summary[molecule][idx].loc[mol]) # mean across available conditions
                par2 = np.nanstd(summary[molecule][idx].loc[mol]) # standard deviation across available conditions
                add = add.append({'parameters':str('c_%s,%s,%s'%(mol,cond,molecule)),'species':cond,'speciestype':molecule,'distribution':'norm','par1':par1,'par2':par2,'ispar':False},ignore_index=True) # add missing values as extra parameters
    return add

#%% Build parameter priors
# For each of the parameters, define the prior/proposal distribution needed for MCMC.
# Inputs: dataframe with parameters, summary generated in define_reactions, the 
# stoichiometric model, and candidate dataframe.
        
def build_priors(param, idx, summary, model, priorKeq=False, candidates=None, sampleNaN=True):
    reaction = model.reactions.get_by_id(summary['rxn_id'][idx])
    distribution, par1, par2,cand,candtype = ([] for i in range(5))
    ispar = np.array([True]*param.shape[0]) # to distinguish between real parameters and sampled NaN
    for i,par in enumerate(param['parameters']):
        if param['speciestype'][i] == 'met': # if species is a metabolite, build priors from metabolomics data
            distribution.append('unif') # log-uniform distribution between median +/-15
            if any(param['species'][i] in s for s in [summary['reactant'][idx].index.values]): # for substrates
                par1.append(-15.0+np.log2(np.nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+np.log2(np.nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
            elif any(param['species'][i] in s for s in [summary['product'][idx].index.values]): # for products
                par1.append(-15.0+np.log2(np.nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+np.log2(np.nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
            elif (candidates is not None): # for candidates
                if list(candidates.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
                           'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']:
                    if (isinstance(candidates['act_coli'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['act_coli'][idx].index.values])): # for activators E. coli
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['act_coli'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['act_coli'][idx].loc[param['species'][i]].values)))
                        cand.append(param['species'][i]); candtype.append('act_coli')
                    elif (isinstance(candidates['inh_coli'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['inh_coli'][idx].index.values])): # for inhibitors E. coli
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['inh_coli'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['inh_coli'][idx].loc[param['species'][i]].values)))
                        cand.append(param['species'][i]); candtype.append('inh_coli')
                    elif (isinstance(candidates['act_other'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['act_other'][idx].index.values])): # for activators
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['act_other'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['act_other'][idx].loc[param['species'][i]].values)))
                        cand.append(param['species'][i]); candtype.append('act_other')
                    elif (isinstance(candidates['inh_other'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['inh_other'][idx].index.values])): # for inhibitor
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['inh_other'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['inh_other'][idx].loc[param['species'][i]].values)))
                        cand.append(param['species'][i]); candtype.append('inh_other')
                else: # for candidates when systematically analyzed directly from metabolites dataframe
                    cond = summary['enzyme'][idx].columns.values
                    par1.append(-15.0+np.log2(np.nanmedian(candidates.loc[param['species'][i][:-2],cond].values)))
                    par2.append(15.0+np.log2(np.nanmedian(candidates.loc[param['species'][i][:-2],cond].values)))
                    cand.append(param['species'][i])
        elif param['speciestype'][i] == 'K_eq': # if the parameter is an equilibrium constant
            Q_r = 1 # calculate Q_r in all conditions
            for subs in list(summary['reactant'][idx].index):
                Q_r /= (summary['reactant'][idx].loc[subs].values)**abs(reaction.get_coefficient(subs))
            products = list(summary['product'][idx].index.values)
            if products:
                for prod in products:
                    Q_r *= (summary['product'][idx].loc[prod].values)**abs(reaction.get_coefficient(prod))
                if priorKeq==False: # log-uniform distribution around meadian +/-20
                    distribution.append('unif')
                    par1.append(-20.0+np.log2(np.nanmedian(Q_r)))
                    par2.append(20.0+np.log2(np.nanmedian(Q_r)))
                else: # if wanted, sample Keq around a normal prior based on BiGG database
                    priorKeqs = pd.read_csv('Keq_bigg.csv',index_col='bigg_id')
                    bools = np.array(list(isinstance(x,str) for x in list(priorKeqs.index.values)))
                    if any(summary['rxn_id'][idx] in s for s in list(priorKeqs.loc[bools].index)):
                        distribution.append('norm')
                        par1.append(float(priorKeqs.loc[summary['rxn_id'][idx],'Keq']))
                        par2.append(float(priorKeqs.loc[summary['rxn_id'][idx],'stdev']))
                    else:
                        print('No prior Keq value for reaction %s' % summary['rxn_id'][idx])
        elif param['speciestype'][i] == 'hill': # log-uniform distribution for Hill coefficient
            distribution.append('unif')
            par1.append(-3)
            par2.append(3)
    param['distribution'] = pd.Series(distribution, index=param.index)
    param['par1'] = pd.Series(par1, index=param.index)
    param['par2'] = pd.Series(par2, index=param.index)
    param['ispar'] = pd.Series(ispar, index=param.index)
    
    # Build priors for sampled NaN values
    if sampleNaN:
        add = pd.DataFrame(columns=param.columns)
        add = fill_nan(add,idx,summary,'reactant') # sample NaN in substrates
        add = fill_nan(add,idx,summary,'product') # sample NaN in products
        add = fill_nan(add,idx,summary,'enzyme') # sample NaN in enzymes
        add = fill_nan(add,idx,summary,'flux') # sample NaN in fluxes
        if (candidates is not None): # sample NaN in candidates
            if list(candidates.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
                       'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']: # from candidates dataframe
                for i,ca in enumerate(cand):
                    isnan = np.isnan(candidates[candtype[i]][idx].loc[ca]) # find nan
                    if any(isnan):
                        for j,cond in enumerate(candidates[candtype[i]][idx].columns[isnan]):
                            par1 = np.nanmedian(candidates[candtype[i]][idx].loc[ca]) # mean across available conditions
                            par2 = np.nanstd(candidates[candtype[i]][idx].loc[ca]) # standard deviation across available conditions
                            add = add.append({'parameters':str('c_%s,%s,%s'%(ca,cond,candtype[i])),'species':cond,'speciestype':candtype[i],'distribution':'norm','par1':par1,'par2':par2,'ispar':False},ignore_index=True)
            else: # if candidates assessed systematically, directly from metabolites dataframe
                for i,ca in enumerate(cand):
                    isnan = np.isnan(candidates.loc[ca[:-2]]) # find nan
                    if any(isnan):
                        for j,cond in enumerate(candidates.columns[isnan]):
                            par1 = np.nanmedian(candidates.loc[ca[:-2]]) # mean across available conditions
                            par2 = np.nanstd(candidates.loc[ca[:-2]]) # standard deviation across available conditions
                            add = add.append({'parameters':str('c_%s,%s'%(ca,cond)),'species':cond,'speciestype':None,'distribution':'norm','par1':par1,'par2':par2,'ispar':False},ignore_index=True)
        param = param.append(add,ignore_index=True) # add sampled NaN in parameters dataframe
    return param

#%% Draw parameters
# From the prior distribution, update those parameters that are present in ‘updates’.
# Inputs: parameter indeces to update within a list, parameter dataframe with priors, current values.
    
def draw_par(update, parameters, current):
    draw = list(current)
    for par in update: # update parameters in update list
        if parameters['distribution'][par]=='unif': # for uniform priors
            draw[par] = 2**uniform(parameters['par1'][par],parameters['par2'][par])
        elif parameters['distribution'][par]=='norm': # for normal priors
            draw[par] = np.abs(normal(parameters['par1'][par],parameters['par2'][par])) # abs to avoid sampling negative values of sampled NaN
        else:
            print('Invalid distribution')
    return draw

#%% Retrieve omics data
# Generate list with data from the summary and candidates dataframes.
# Inputs: recations idx, summary dataframe, MM equations, candidates dataframe and regulators list.
    
def retrieve_omics_data(idx,summary,equations,candidates=None,regulator=None):
    occu = equations['occu']
    vbles = []
    vbles_vals = []
    for sub in list(summary['reactant'][idx].index): # retrieve substrate data from summary dataframe
        vbles.append('c_'+sub)
        vbles_vals.append(summary['reactant'][idx].loc[sub].values)
    for prod in list(summary['product'][idx].index): # retrieve product data from summary dataframe
        vbles.append('c_'+prod)
        vbles_vals.append(summary['product'][idx].loc[prod].values)
    if regulator:
        for reg in regulator: # retrieve candidate data
            reg = reg[4:]
            if list(candidates.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
                       'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']: # from candidate dataframe if not systematic
                if (isinstance(candidates['act_coli'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['act_coli'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg)
                    vbles_vals.append(candidates['act_coli'][idx].loc[reg].values)
                elif (isinstance(candidates['inh_coli'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['inh_coli'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg)
                    vbles_vals.append(candidates['inh_coli'][idx].loc[reg].values)
                elif (isinstance(candidates['act_other'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['act_other'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg)
                    vbles_vals.append(candidates['act_other'][idx].loc[reg].values)
                elif (isinstance(candidates['inh_other'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['inh_other'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg)
                    vbles_vals.append(candidates['inh_other'][idx].loc[reg].values)
            elif not (any('c_'+reg in s for s in vbles)): # directly from metabolites dataframe if systematic
                cond = summary['enzyme'][idx].columns.values # take same conditions as in summary
                vbles.append('c_'+reg)
                vbles_vals.append(candidates.loc[reg[:-2],cond].values)
    param = list(set(re.findall('(?<![A-Za-z0-9_])[Kn]_[A-Za-z0-9_]+_c|K_eq|K_reg_[A-Za-z0-9_]+_c',str(occu)))) # retrieve list of parameters from MM equation using regular expression
    vbles.extend(param) # add parameters in list of variables
    f = sym.lambdify(vbles, occu) # create function with the expression in 'occu' taking the defined variables
    return param,vbles_vals,f

#%% Calculate likelihood
# Calculate the likelihood given the flux point estimate or the lower and upper bounds of flux variability analysis.
# Inputs: parameter dataframe, current values, summary as generated in define_reactions, 
# equations, and candidates.
    
def calculate_lik(idx,parameters, current, summary, vbles_vals, param, f):
    enz = summary['enzyme'][idx].values # enzyme abundancies
    ncond = enz.shape[1] # number of conditions
    vbles_vals = list(vbles_vals)
    for par in param: # the function f needs to be assessed per each condition, with parameters being the same across all
        rep_par = np.repeat(current[parameters['parameters'].values==par],ncond) # repeat parameters as many times as conditions
        vbles_vals.append(rep_par)
    
    flux = summary['flux'][idx] # observed fluxes
    bool_occu = (np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0))==max(np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0)) # conditions with no NaN across all 'occu' variables
    bool_enz = (np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),list(enz))),1),0))==max(np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),list(enz))),1),0)) # conditions with no NaN in enzyme abundancies
    bool_all = ((np.isnan(flux.iloc[0,:])==0).values.reshape(ncond,)&(bool_occu)&(bool_enz)) # condtions with no NaN across all 'occu' variables, enzyme abundancies and fluxes
    vbles_vals = list(map(lambda x: x[bool_all],vbles_vals)) # keep only variable values for conditions in bool_all
    ncond = np.sum(bool_all) # number of conditions in bool_all
    pred_occu = f(*vbles_vals) # evaluate MM function for all conditions in bool_all to obtain the 'occu' values
    if flux.shape[0] == 1: # fluxes as point estimates
        kcat, residual = nnls(np.transpose(pred_occu*enz[:,bool_all]), flux.loc[:,bool_all].values.reshape(ncond)) # given the 'occu' values, determine kcat using NNLS
        pred_flux = np.sum(kcat*np.transpose(enz[:,bool_all]),1)*pred_occu # calculate predicted flux given 'occu', kcats and enzyme abundancies
        var = residual**2/(ncond) # calculate the variance of the likelihood (the same for all conditions)
        likelihood = norm.pdf(flux.loc[:,bool_all].values, pred_flux, np.sqrt(var)) # calculate likelihood
        if isinstance(likelihood,np.ndarray): # reformat likelihood
            likelihood = likelihood[0]
        return np.nansum(np.log(likelihood)),kcat,pred_flux,np.log(likelihood),bool_all

    elif flux.shape[0] == 2: # min/max range of fluxes
        ave_flux = np.mean(flux.values,0) # calculate average flux
        kcat, residual = nnls(np.transpose(pred_occu*enz[:,bool_all]), ave_flux[bool_all]) # given the 'occu' values, determine kcat using NNLS
        pred_flux = np.sum(kcat*np.transpose(enz[:,bool_all]),1)*pred_occu # calculate predicted flux given 'occu', kcats and enzyme abundancies
        if np.sum(ave_flux)!=0.0: # calculate likelihood only if flux is not zero for at least one condition
            var = residual**2/(ncond) # calculate the variance of the likelihood (the same for all conditions)
            likelihood = (norm.cdf(flux.loc['max',bool_all].values, pred_flux, np.sqrt(var)) - norm.cdf(flux.loc['min',bool_all].values, pred_flux, np.sqrt(var)))/(flux.loc['max',bool_all].values - flux.loc['min',bool_all].values) # calculate likelihood
            return np.nansum(np.log(likelihood)),kcat,pred_flux,np.log(likelihood),bool_all
        else:
            return None,None,None,None,None # Quit the evaluation of this expression

#%% Reconstruct summary and candidates
# Add sampled values for NaN into summary and candidates.
# Inputs: current par, array with parameter names, summary, candidates.
            
def add_sampled(idx, current, parameters, summary, candidates=None):
    met_bool = np.array([',' in s for s in parameters]) # find which rows are not real parameters, but sampled NaN values
    current = np.array(current); current = current[met_bool]
    for i, spe in enumerate(parameters[met_bool]):
        specie = re.split(',', spe) # split parameter name by the comma
        if len(specie)==2: # in systematic assessment of candidates, species contains: 0.molecule name,1.condition
            candidates.loc[specie[0][2:-2],specie[1]] = current[i]
        elif any([specie[2] in s for s in ['reactant','product','enzyme','flux']]): # for species in summary dataframe: 0.molecule name,1.condition,2.molecule type
            mol_df = summary[specie[2]][idx].copy()
            mol_df.loc[specie[0][2:],specie[1]] = current[i]
            summary[specie[2]][idx] = mol_df
        elif any([specie[2] in s for s in ['act_coli', 'act_other','inh_coli', 'inh_other']]): # for species in candidate dataframe: 0.molecule name,1.condition,2.regulatory type
            mol_df = candidates[specie[2]][idx].copy()
            mol_df.loc[specie[0][2:],specie[1]] = current[i]
            candidates[specie[2]][idx] = mol_df
    return summary,candidates

#%% Fit reaction equation using MCMC-NNLS
# Sample posterior distribution Pr(Ω|M,E,jF) using MCMC-NNLS.
# Inputs: markov parameters (fraction of samples that are reported, how many samples are 
# desired, how many initial samples are skipped), parameters table with priors, 
# summary as generated in define_reactions, equations, and candidates.
    
def fit_reaction_MCMC(idx, markov_par, parameters, summary, equations,candidates=None,regulator=None,sampleNaN=True):
    print('Running MCMC-NNLS for reaction %d... Candidate regulator: %s' % (idx,regulator))
    colnames = list(parameters['parameters'].values) # name of parameters
    colnames.extend(re.findall('K_cat_[a-zA-Z0-9_]+', str(equations['vmax']))+['likelihood','pred_flux','lik_cond']) # add extra columns
    track = pd.DataFrame(columns=colnames)
    current_pars = [None] * parameters.shape[0]
    current_pars = draw_par([p for p in range(parameters.shape[0])], parameters, current_pars) # initialize parameter set
    summary_cp=summary.copy() # always work on copies, since changes on dataframe are recurrent. Avoid changes on the original df when changing the copy
    if candidates is not None:
        candidates_cp=candidates.copy()
    else: 
        candidates_cp = None    
    if (sampleNaN and parameters.loc[parameters['ispar']==False].empty==0):
        summary_cp,candidates_cp = add_sampled(idx, current_pars, parameters['parameters'].values, summary_cp, candidates_cp)
    param,vbles_vals,f = retrieve_omics_data(idx,summary_cp,equations,candidates_cp,regulator)
    current_lik,cur_kcat,cur_pred_flux,cur_lik_cond,bool_all = calculate_lik(idx, parameters.loc[parameters['ispar']], \
                                np.array(current_pars)[parameters['ispar']==True], summary_cp, vbles_vals, param, f)
    if current_lik is None:
        print('ERROR: FVA fluxes are completely unconstrained.')
        return None,None # Quit the evaluation of this expression
    else:
        for i in range(markov_par['burn_in']+markov_par['nrecord']*markov_par['freq']): # total number of iterations
            for p in range(parameters.shape[0]): # loop through all parameters
                proposed_pars = draw_par([p], parameters, current_pars)
                if (sampleNaN and parameters.loc[parameters['ispar']==False].empty==0):
                    summary_cp,candidates_cp = add_sampled(idx, proposed_pars, parameters['parameters'].values, summary_cp, candidates_cp)
                    param,vbles_vals,f = retrieve_omics_data(idx,summary_cp,equations,candidates_cp,regulator)
                proposed_lik,pro_kcat,pro_pred_flux,pro_lik_cond,bool_all = calculate_lik(idx, parameters.loc[parameters['ispar']], \
                                    np.array(proposed_pars)[parameters['ispar']==True], summary_cp, vbles_vals, param, f)
                if ((uniform(0,1) < np.exp(proposed_lik)/(np.exp(proposed_lik)+np.exp(current_lik))) or \
                    (proposed_lik > current_lik) or ((proposed_lik==current_lik)and(proposed_lik==-np.inf))): # accept new parameter set if likelihood improves or decreases slightly
                    current_pars = proposed_pars
                    cur_kcat = pro_kcat
                    cur_pred_flux = pro_pred_flux
                    current_lik = proposed_lik
                    cur_lik_cond = pro_lik_cond
            if (i > markov_par['burn_in']):
                if ((i-markov_par['burn_in'])%markov_par['freq'])==0: # record parameter set at the defined frequency
                    add_pars = list(current_pars)
                    add_pars.extend(cur_kcat); add_pars.append(current_lik); add_pars.append(cur_pred_flux); add_pars.append(cur_lik_cond)
                    track = track.append(pd.DataFrame([add_pars],columns=colnames))
        track.reset_index(drop=True,inplace=True)
        return track,bool_all

#%% Calculate uncertainty of prediction
# Based on the variances of species and predicted fluxes, estimate the uncertainty 
# of the prediction using the multivariate delta method.
# Inputs: idx, equation, parameter dataframe, summary and candidates dataframes, regulator, candidates_sd
# in case all metabolites are being tested.

def cal_uncertainty(idx, expr, parameters, summary, candidates=None, regulator=None, candidates_sd=None):
    vbles,vbles_vals,sd_vals,species = ([] for l in range(4))
    ncond = summary['enzyme'][idx].shape[1] # number of conditions
    for par in range(parameters.shape[1]): # retrieve optimal parameter values
        vbles.append(parameters.columns[par])
        rep_par = np.repeat(parameters.iloc[0,par],ncond)
        vbles_vals.append(rep_par)
    for sub in list(summary['reactant'][idx].index): # retrieve substrate abundancies
        vbles.append('c_'+sub); species.append('c_'+sub)
        vbles_vals.append(summary['reactant'][idx].loc[sub].values)
        sd_vals.append(summary['reactant_sd'][idx].loc[sub].values)
    for prod in list(summary['product'][idx].index): # retrieve product abundancies
        vbles.append('c_'+prod); species.append('c_'+prod)
        vbles_vals.append(summary['product'][idx].loc[prod].values)
        sd_vals.append(summary['product_sd'][idx].loc[prod].values)
    for enz in list(summary['enzyme'][idx].index): # retrieve enzyme abundancies
        vbles.append('c_'+enz); species.append('c_'+enz)
        vbles_vals.append(summary['enzyme'][idx].loc[enz].values)
        sd_vals.append(summary['enzyme_sd'][idx].loc[enz].values)
    if regulator: # retrieve regulator abundancies
        for reg in regulator:
            reg = reg[4:]
            if list(candidates.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
                       'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']: # from candidate dataframe
                if (isinstance(candidates['act_coli'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['act_coli'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg); species.append('c_'+reg)
                    vbles_vals.append(candidates['act_coli'][idx].loc[reg].values)
                    sd_vals.append(candidates['act_coli_sd'][idx].loc[reg].values)
                elif (isinstance(candidates['inh_coli'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['inh_coli'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg); species.append('c_'+reg)
                    vbles_vals.append(candidates['inh_coli'][idx].loc[reg].values)
                    sd_vals.append(candidates['inh_coli_sd'][idx].loc[reg].values)
                elif (isinstance(candidates['act_other'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['act_other'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg); species.append('c_'+reg)
                    vbles_vals.append(candidates['act_other'][idx].loc[reg].values)
                    sd_vals.append(candidates['act_other_sd'][idx].loc[reg].values)
                elif (isinstance(candidates['inh_other'][idx],pd.DataFrame)) and \
                (any(reg in s for s in [candidates['inh_other'][idx].index.values])) and not (any('c_'+reg in s for s in vbles)):
                    vbles.append('c_'+reg); species.append('c_'+reg)
                    vbles_vals.append(candidates['inh_other'][idx].loc[reg].values)
                    sd_vals.append(candidates['inh_other_sd'][idx].loc[reg].values)
            elif not (any('c_'+reg in s for s in vbles)): # directly from metabolites dataframe if systematic assessment
                cond = summary['flux'][idx].columns.values
                vbles.append('c_'+reg); species.append('c_'+reg)
                vbles_vals.append(candidates.loc[reg[:-2],cond].values)
                sd_vals.append(candidates_sd.loc[reg[:-2],cond].values)
    
    flux = summary['flux'][idx] # retrieve fluxes
    bool_occu = (np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0))==max(np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0)) # remove conditions with NaN
    bool_all = ((np.isnan(flux.iloc[0,:])==0).values.reshape(ncond,)&(bool_occu))
    vbles_vals = list(map(lambda x: x[bool_all],vbles_vals)) # keep only condition without NaNs
    ncond = np.sum(bool_all) # number of remaining conditions
    grads = np.zeros((ncond,len(species))) # gradients
    expr = expr['vmax']*expr['occu'] # MM expression
    uncertainty = np.zeros((1,ncond))
    cov_matrix = np.zeros((len(species),len(species))) # covariance matrix
    for m,spe1 in enumerate(species): # loop through metabolites and enzymes to determine gradients
        gradient = sym.diff(expr,species[m])
        f = sym.lambdify(vbles, gradient)
        grads[:,m] = f(*vbles_vals)
    for i in range(ncond):
        for m,spe1 in enumerate(species): # loop through metabolites and enzymes to determine covariances
            for n,spe2 in enumerate(species):
                if n>=m: # cov(m,n)=cov(n,m)
                    cov_matrix[m,n] = sd_vals[m][i]*sd_vals[n][i]
                    cov_matrix[n,m] = sd_vals[n][i]*sd_vals[m][i]
        uncertainty[0,i] = np.dot(np.dot(grads[i],cov_matrix),np.transpose(grads[i])) # multivariate delta method
    return uncertainty
#%% Fit reaction equations
# Run all required functions as a block to fit predicted to measured flux.
# Inputs: summary dataframe, stoichiometric model, markov parameters, and candidates dataframe (optional).
    
def fit_reactions(summary,model,markov_par,candidates=None,candidates_sd=None,priorKeq=False,maxreg=1,coop=False,sampleNaN=True):
    results = pd.DataFrame(columns=['idx','reaction','rxn_id','regulator','equation',\
                                    'meas_flux','pred_flux','best_fit','best_lik','lik_cond'])
    for idx in list(summary.index):
        expr,parameters,regulator = write_rate_equations(idx,summary,model,candidates,maxreg,coop)
        for i in range(len(expr)):
            parameters[i] = build_priors(parameters[i],idx,summary,model,priorKeq,candidates,sampleNaN)
            track,bool_all = fit_reaction_MCMC(idx,markov_par,parameters[i],summary,expr[i],candidates,regulator[i],sampleNaN)
            if track is None:
                continue # Quit the evaluation of this expression
            else:
                max_lik = max(track['likelihood'].values) # best likelihood
                max_par = track[track['likelihood'].values==max_lik] # parameter set(s) with the optimal likelihood
                par_bool = [',' not in s for s in max_par.columns]; par_bool[-3:]=[False]*3 # find real parameters
                uncertainty = cal_uncertainty(idx, expr[i], max_par.loc[:,par_bool],summary,candidates,regulator[i],candidates_sd)
                add = {'idx':idx,'reaction':summary['reaction'][idx],'rxn_id':summary['rxn_id'][idx],\
                       'regulator':regulator[i],'equation':(expr[i]['vmax']*expr[i]['occu']),'meas_flux':summary['flux'][idx].loc[:,bool_all],\
                       'pred_flux':max_par.iloc[:,-2].values,'uncertainty':uncertainty[0],'best_fit':max_par.iloc[:,:-3],'best_lik':max_lik,\
                       'lik_cond':max_par.iloc[:,-1].values[0]}
                results = results.append([add])
    results = results.sort_values(by='best_lik') # sort results by the likelihood
    results.reset_index(drop=True,inplace=True)
    return results

#%% Validate results
# Run likelihood ratio test and Bayesian a posteriori probability given a gold standard of regulators.
# Inputs: results dataframe.
    
def validate(results, gold_std=None, fullreg=None):
    noreg = results.loc[results['regulator']==''].reset_index(drop=True) # results of unregulated models
    noreg['pvalue']= np.ones((noreg.shape[0],1)) # set pval to 1 for unregulated models
    validation = noreg[['rxn_id','regulator','best_lik','pvalue']]
    ncond = list(map(lambda x: len(x),list(noreg['lik_cond'].values))) # number of conditions in each model
    npar = [x.loc[:,[',' not in s for s in list(x.columns)]].shape[1] for x in list(noreg['best_fit'])] # number of parameters in each model
    validation.reset_index(drop=True,inplace=True)
    for i,rxn in enumerate(list(noreg['rxn_id'].values)): # loop through all reactions
        rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True) # results of regulated models for that reaction
        if rxn_results.empty==0:
            rxn_results['pvalue']= np.ones((rxn_results.shape[0],1))
            for j,reg in enumerate(list(rxn_results['regulator'].values)):
                ratio = 2*(rxn_results['best_lik'].iloc[j]-noreg['best_lik'].iloc[i]) # likelihood ratio test
                p = chi2.sf(ratio, len(reg)) # calculate pvalue
                rxn_results.loc[j,'pvalue']=p
                ncond.append(len(rxn_results['lik_cond'].iloc[j]))
                npar.append(rxn_results['best_fit'].iloc[j].shape[1])
            validation = validation.append(rxn_results[['rxn_id','regulator','best_lik','pvalue']],ignore_index=True)
    
    if gold_std is not None: # logistic regression using gold standard as true regulation
        reactions = gold_std.index.values # reactions in gold standard
        regs_coli = fullreg.loc[reactions] # known regulators in E. coli
        regs_unique = regs_coli.drop_duplicates()
        n_tested = np.zeros([regs_unique.shape[0],1])
        n_ecoli = np.zeros([regs_unique.shape[0],1])
        outcome = np.zeros([regs_unique.shape[0],])
        for i,rxn in enumerate(list(regs_unique.index.values)): # loop through E. coli regulators
            n_tested[i,0] = np.log2(regs_unique.loc[rxn].shape[0]) # total number of regulators for that reaction
            n_ecoli[i,0] = np.sum(regs_coli['metab'].loc[rxn]==regs_unique['metab'].iloc[i]) # number of times regulator i has been reported in E. coli
            if any(regs_unique['metab'].iloc[i] in s for s in list(gold_std['metabolite'].loc[rxn])):
                outcome[i] = 1 # regulators in gold standard are considered true, and all other ones as false
        income = np.concatenate([n_tested,n_ecoli],1)
        logreg = LogisticRegression()
        logreg.fit(income,outcome) # train logistic model
        
        reactions = [s.lower() for s in list(noreg['rxn_id'].values)] # reactions to test
        regs_coli = fullreg.loc[reactions] # E. coli regulators corresponding to these reactions
        regs_unique = regs_coli.reset_index().drop_duplicates().set_index('rxn_id')
        validation2 = validation.loc[validation['regulator']!=''] # regulated models
        n_tested = np.zeros([validation2.shape[0],1])
        n_ecoli = np.zeros([validation2.shape[0],1])
        for i,rxn in enumerate(list(validation2['rxn_id'].values)): # calculate incomes of logistic model
            n_tested[i,0] = np.log2(regs_unique.loc[rxn.lower()].shape[0])
            n_ecoli[i,0] = np.sum([np.sum(regs_coli['metab'].loc[rxn.lower()]==r[4:-2]) for r in validation2['regulator'].iloc[i]])
        test = np.concatenate([n_tested,n_ecoli],1)
        prior_reg = logreg.predict_proba(test) # predict outcome based on logistic model
        prior_all = np.concatenate([np.ones([noreg.shape[0],]),prior_reg[:,0]])
        validation['posteriori'] = np.log2(prior_all*(2**validation['best_lik'].values)) # calculate a posteriori probaibily of regulator
        
    aic = np.zeros([validation.shape[0],1]) # calculate Akaike information criiterion
    for i,rxn in enumerate(list(validation['rxn_id'].values)):
        if gold_std is None:
            aic[i,0] = 2*npar[i]-2*validation['best_lik'].iloc[i]
        else:
            aic[i,0] = 2*npar[i]-2*validation['posteriori'].iloc[i]
    validation['AIC'] = aic
    validation['ncond'] = ncond
    return validation

#%% Validate results by condition
# Run likelihood ratio test in a condition-specific manner and weight regulation scores according to this.
# Inputs: results, summary and candidates dataframes.
    
def validate_bycond(results,summary=None,candidates=None):
    noreg = results.loc[results['regulator']==''].reset_index(drop=True) # results of unregulated models
    ncond = list(map(lambda x: len(x),list(noreg['lik_cond'].values))) # number of conditions in each model
    npar = [x.loc[:,[',' not in s for s in list(x.columns)]].shape[1] for x in list(noreg['best_fit'])] # number of parameters in each model
    noreg['pvalue'] = list(map(lambda i: np.ones((1,ncond[i]))[0],np.arange(noreg.shape[0])))
    noreg['elasticity'] = list(map(lambda i: np.zeros((1,ncond[i]))[0],np.arange(noreg.shape[0])))
    noreg['pvalue_weighted'] = np.ones((noreg.shape[0],1))
    noreg['elasticity_weighted'] = np.zeros((noreg.shape[0],1))
    noreg['lik_weighted'] = noreg['best_lik'].copy()
    validation = noreg[['rxn_id','regulator','lik_cond','lik_weighted','pvalue','pvalue_weighted','elasticity','elasticity_weighted']]
    validation.reset_index(drop=True,inplace=True)
    for i,rxn in enumerate(list(noreg['rxn_id'].values)): # loop through reactions
        rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True) # results of regulated models of the reaction
        if rxn_results.empty==0:
            pval, pval_weight, elas, elas_weight, lik_weight = ([] for i in range(5))
            idx = int(summary.loc[summary['rxn_id']==rxn].index.values)
            for j,reg in enumerate(list(rxn_results['regulator'].values)):
                bool_cond = np.array(list(map(lambda x: any(x in s for s in list(rxn_results.loc[j,'meas_flux'].columns)),list(noreg.loc[i,'meas_flux'].columns)))) # boolean of available conditions
                cond = list(rxn_results.loc[j,'meas_flux'].columns) # available conditions
                ratio = 2*(rxn_results['lik_cond'].iloc[j]-noreg['lik_cond'].iloc[i][bool_cond]) # likelihood ratio test
                p = chi2.sf(ratio, len(reg)) # calculate pval
                pval.append(p)
                weights = (np.log(p)/np.nansum(np.log(p))) # calculate condition weights based on pval improvement
                pval_weight.append(np.nansum(p*weights)) # weighted pval
                lik_weight.append(np.nansum(rxn_results['lik_cond'].iloc[j]*weights)*len(cond)) # weighted likelihood
                ncond.append(len(cond)) # number of available conditions
                par_bool = [',' not in s for s in rxn_results['best_fit'].iloc[j].columns] # identify real parameters
                npar.append(rxn_results['best_fit'].iloc[j].loc[:,par_bool].shape[1]) # number of parameters
                if (summary is not None) and (candidates is not None): # calculate elasticities
                    summary_cp=summary.copy(); candidates_cp=candidates.copy()
                    parameters = rxn_results['best_fit'].iloc[j].columns
                    if any([',' in s for s in parameters]): # add sampled NaN
                        summary_cp,candidates_cp = add_sampled(idx,list(rxn_results['best_fit'].iloc[j].iloc[0,:]), parameters, summary_cp, candidates_cp)
                    vbles,vbles_vals = ([] for l in range(2))
                    for par in range(npar[-1]): # retrieve parameter values
                        vbles.append(rxn_results['best_fit'].iloc[j].loc[:,par_bool].columns[par])
                        rep_par = np.repeat(rxn_results['best_fit'].iloc[j].iloc[0,par],ncond[-1])
                        vbles_vals.append(rep_par)
                    for sub in list(summary_cp.loc[idx,'reactant'].index): # retrieve substrate values
                        vbles.append('c_'+sub)
                        vbles_vals.append(summary_cp.loc[idx,'reactant'].loc[sub,cond].values)
                    for prod in list(summary_cp.loc[idx,'product'].index): # retrieve product values
                        vbles.append('c_'+prod)
                        vbles_vals.append(summary_cp.loc[idx,'product'].loc[prod,cond].values)
                    for enz in list(summary_cp.loc[idx,'enzyme'].index): # retrieve enzyme values
                        vbles.append('c_'+enz)
                        vbles_vals.append(summary_cp.loc[idx,'enzyme'].loc[enz,cond].values)
                    reg_data = {}
                    for rg in reg: # retrieve regulator values
                        rg = rg[4:]
                        if list(candidates_cp.columns.values)==['act_coli','act_coli_sd', 'act_other', 'act_other_sd',\
                                   'inh_coli','inh_coli_sd', 'inh_other','inh_other_sd']: # from candidates dataframe
                            if (isinstance(candidates_cp.loc[idx,'act_coli'],pd.DataFrame)) and \
                            (any(rg in s for s in [candidates_cp.loc[idx,'act_coli'].index.values])):
                                if not (any('c_'+rg in s for s in vbles)): # avoid readding it if regulator is already a substrate or product
                                    vbles.append('c_'+rg)
                                    vbles_vals.append(candidates_cp.loc[idx,'act_coli'].loc[rg,cond].values)
                                    reg_data[rg] = candidates_cp.loc[idx,'act_coli'].loc[rg,cond].values
                                else:
                                    reg_data[rg] = candidates_cp.loc[idx,'act_coli'].loc[rg,cond].values
                            elif (isinstance(candidates_cp.loc[idx,'inh_coli'],pd.DataFrame)) and \
                            (any(rg in s for s in [candidates_cp.loc[idx,'inh_coli'].index.values])):
                                if not (any('c_'+rg in s for s in vbles)): # avoid readding it if regulator is already a substrate or product
                                    vbles.append('c_'+rg)
                                    vbles_vals.append(candidates_cp.loc[idx,'inh_coli'].loc[rg,cond].values)
                                    reg_data[rg]=candidates_cp.loc[idx,'inh_coli'].loc[rg,cond].values
                                else:
                                    reg_data[rg]=candidates_cp.loc[idx,'inh_coli'].loc[rg,cond].values
                            elif (isinstance(candidates_cp.loc[idx,'act_other'],pd.DataFrame)) and \
                            (any(rg in s for s in [candidates_cp.loc[idx,'act_other'].index.values])):
                                if not (any('c_'+rg in s for s in vbles)): # avoid readding it if regulator is already a substrate or product
                                    vbles.append('c_'+rg)
                                    vbles_vals.append(candidates_cp.loc[idx,'act_other'].loc[rg,cond].values)
                                    reg_data[rg]=candidates_cp.loc[idx,'act_other'].loc[rg,cond].values
                                else:
                                    reg_data[rg]=candidates_cp.loc[idx,'act_other'].loc[rg,cond].values
                            elif (isinstance(candidates_cp.loc[idx,'inh_other'],pd.DataFrame)) and \
                            (any(rg in s for s in [candidates_cp.loc[idx,'inh_other'].index.values])):
                                if not (any('c_'+rg in s for s in vbles)): # avoid readding it if regulator is already a substrate or product
                                    vbles.append('c_'+rg)
                                    vbles_vals.append(candidates_cp.loc[idx,'inh_other'].loc[rg,cond].values)
                                    reg_data[rg]=candidates_cp.loc[idx,'inh_other'].loc[rg,cond].values
                                else:
                                    reg_data[rg]=candidates_cp.loc[idx,'inh_other'].loc[rg,cond].values
                        elif not (any('c_'+rg in s for s in vbles)): # directly from metabolites dataframe if systematic assessment
                            vbles.append('c_'+rg)
                            vbles_vals.append(candidates_cp.loc[rg[:-2],cond].values)
                            reg_data[rg]=candidates_cp.loc[rg[:-2],cond].values
                        else:
                            reg_data[rg]=candidates_cp.loc[rg[:-2],cond].values

                    expr = rxn_results['equation'].iloc[j]
                    f = sym.lambdify(vbles, expr)
                    f_res = f(*vbles_vals) # calculate predicted fluxes 
                    el = []
                    for n,rg in enumerate(reg):
                        gradient = sym.diff(expr,'c_'+rg[4:])
                        g = sym.lambdify(vbles, gradient)
                        g_res = g(*vbles_vals) # calculate gradients with respect to regulators
                        el.append(g_res*reg_data[rg[4:]]/f_res)
                    elas.append(el)
                    elas_weight.append(np.nansum(el*weights,1)) # weighted elasticities
                else:
                    elas.append(np.zeros((1,ncond[-1])))
                    elas_weight.append(0)
            rxn_results['lik_weighted']=lik_weight
            rxn_results['pvalue']=pval
            rxn_results['pvalue_weighted']=pval_weight
            rxn_results['elasticity']=elas
            rxn_results['elasticity_weighted']=elas_weight
            validation = validation.append(rxn_results[['rxn_id','regulator','lik_cond','lik_weighted','pvalue','pvalue_weighted','elasticity','elasticity_weighted']],ignore_index=True)
            
    aic = []; aic_weight = []
    for i,rxn in enumerate(list(validation['rxn_id'].values)): # calculate Akaike information criterion
        aic.append(2*npar[i]-2*validation['lik_cond'].iloc[i])
        if validation['pvalue'].iloc[i][0] != 1.0:
            weights = (np.log(validation['pvalue'].iloc[i])/np.nansum(np.log(validation['pvalue'].iloc[i])))
        else:
            weights = np.ones((1,len(validation['pvalue'].iloc[i])))/len(validation['pvalue'].iloc[i])
        aic_weight.append(np.nansum(aic[-1]*weights)) # weighted AIC
    validation['AIC'] = aic
    validation['AIC_weighted'] = aic_weight
    validation['ncond'] = ncond
    
    val_results = pd.DataFrame(columns=results.columns)
    for rxn in list(results['rxn_id'].drop_duplicates()): # keep only results that improve AIC
            rxn_results = results.loc[results['rxn_id']==rxn]
            rxn_val = validation.loc[validation['rxn_id']==rxn]
            aic_noreg = rxn_val.loc[(rxn_val['regulator']==''),'AIC_weighted'].values[0]
            idx = rxn_val.loc[(rxn_val['AIC_weighted']<=aic_noreg),'regulator']
            bools =[any([i==s for s in idx]) if ((i!='')and not(idx.empty)) else False if idx.empty else True for i in rxn_results['regulator']]
            red = rxn_results.loc[bools]
            addcol = ['lik_weighted','AIC','AIC_weighted','elasticity','elasticity_weighted']
            add = [rxn_val.loc[[r==s for s in rxn_val['regulator']],addcol] for r in red['regulator']]
            add = pd.concat(add)
            add = add.set_index(red.index)
            red = pd.concat([red, add], axis=1)
            val_results = val_results.append(red)
    val_results = val_results.sort_values(by='AIC_weighted',ascending=False)
    val_results.reset_index(drop=True,inplace=True)
    
    return validation, val_results

#%% Multi-layer information prioritization score
# Score all candidates according to prior information, binding information from NMR and LiP, and elasticities.
# Inputs: results dataframe and stoichiometric COBRA model.
    
def rankresults(val_results,model):
    # Open files with prior knowledge, LiP-Proteomics, NMR, and mapping info.
    nmr = pd.read_excel('171208_MD_NMR_PMIs.xlsx',index_col=0)
    lim_prot = pd.read_csv('limited_proteolysis_Elad.csv',index_col='ProteinID')
    reg_coli = pd.read_csv("SMRN.csv",index_col="rxn_id")
    reg_other = pd.read_excel("SMRN_allorgs.xlsx",index_col='EC_number')
    mapping = pd.read_table('bigg_models_reactions.txt',index_col='bigg_id')
    prot_map = pd.read_table('ECOLI_83333_idmapping.dat',header=None,index_col=0)
    
    shortlisting = pd.DataFrame(index=val_results['rxn_id'],columns=['regulator','deltaAIC','AIC','lik_cond','lik_weighted','elasticity','elasticity_weighted','in_ecoli','in_allorgs','in_nmr','in_limpro','score'])
    shortlisting['in_nmr'] = np.zeros([shortlisting.shape[0],1])
    shortlisting['in_limpro'] = np.zeros([shortlisting.shape[0],1])
    shortlisting['score'] = np.zeros([shortlisting.shape[0],1])
    shortlisting['in_allorgs'] = np.zeros([shortlisting.shape[0],1])
    for n,rxn in enumerate(shortlisting.index): # loop through models
        rxn_results = val_results.iloc[[n]]
        ec = re.findall('[1-6]+\.[0-9]+\.[0-9]+\.[0-9]+',mapping.loc[rxn,'database_links']) # find corresponding EC numbers
        other_rxn = pd.DataFrame(columns=reg_other.columns)
        for i in ec:
            if i in reg_other.index:
                other_rxn = other_rxn.append(reg_other.loc[ec],ignore_index=True) # extract known regulators in any organism for these EC numbers
        if rxn.lower() in reg_coli.index:
            coli_rxn = reg_coli.loc[[rxn.lower()]] # extract known regulators in E. coli
        else:
            coli_rxn = pd.DataFrame(columns=reg_coli.columns)
        regs = rxn_results['regulator'].values[0] # regulator(s)
        n_other = np.sum(any([np.sum(rg[4:-2]==s for s in list(other_rxn.loc[other_rxn['Mode']=='+','metab'])) if rg[0:3]=='ACT' else np.sum(rg[4:-2]==s for s in list(other_rxn.loc[other_rxn['Mode']=='-','metab'])) for rg in regs])) # is this regulator known in any organism?
        n_coli = np.sum(any([np.sum(rg[4:-2]==s for s in list(coli_rxn.loc[coli_rxn['mode']=='+','metab'])) if rg[0:3]=='ACT' else np.sum(rg[4:-2]==s for s in list(coli_rxn.loc[coli_rxn['mode']=='-','metab'])) for rg in regs])) # is this regulator known in E. coli?
        shortlisting['in_ecoli'].iloc[n]=n_coli
        shortlisting['in_allorgs'].iloc[n]=n_other
        shortlisting['regulator'].iloc[n]=regs
        shortlisting['deltaAIC'].iloc[n]=rxn_results['AIC_weighted'].values[0]-val_results.loc[np.logical_and(val_results['rxn_id']==rxn,val_results['regulator']==''),'AIC_weighted'].values[0] # improvement of AIC with respect to the unregulated model
        shortlisting['AIC'].iloc[n]= rxn_results['AIC'].values[0]
        shortlisting['lik_cond'].iloc[n]= rxn_results['lik_cond'].values[0]
        shortlisting['lik_weighted'].iloc[n]= rxn_results['lik_weighted'].values[0]
        shortlisting['elasticity'].iloc[n]= rxn_results['elasticity'].values[0]
        shortlisting['elasticity_weighted'].iloc[n]= rxn_results['elasticity_weighted'].values[0]
        gene = list(model.reactions.get_by_id(rxn).genes) # b numbers
        prot = [str(prot_map.loc[prot_map.loc[:,2]==g.id].index[0]) for g in gene] # obtain gene names from b numbers
        names = [prot_map.loc[prot_map.iloc[:,0]=='Gene_Name'].loc[p,2].lower() if p in prot_map.loc[prot_map.iloc[:,0]=='Gene_Name'].index else p for p in prot]
        in_nmr = np.nan; in_limpro = np.nan; 
        for r in regs:
            if (r[4:-2] in nmr.columns) and (in_nmr!=True) and (any([g in nmr.index for g in names])): # detect interaction in nmr
                in_nmr = any([g in nmr.loc[np.bool_(nmr.loc[:,r[4:-2]].values)].index for g in names])
            if (r[4:-2] in lim_prot.columns) and (in_limpro!=True) and (any([p in lim_prot.index for p in prot])): # detect interaction in limpro
                in_limpro = any([p in lim_prot.loc[np.bool_(lim_prot.loc[:,r[4:-2]].values)].index for p in prot])
        if in_nmr==False: # nmr score
            shortlisting['in_nmr'].iloc[n] = -1.0
        elif in_nmr==True:
            shortlisting['in_nmr'].iloc[n] = 1.0
        if in_limpro==False: # limpro score
            shortlisting['in_limpro'].iloc[n] = -1.0
        elif in_limpro==True:
            shortlisting['in_limpro'].iloc[n] = 1.0
            
    noregbool = shortlisting['regulator']!='' # unregulated models
    maxelas = [np.max(np.abs(e)) for e in shortlisting.loc[noregbool,'elasticity_weighted']] # absolute values of maximum elasticities
    stdevs_inv = np.array([1/np.std(shortlisting.loc[noregbool,'in_nmr']),1/np.std(shortlisting.loc[noregbool,'in_limpro']),1/np.std(shortlisting.loc[noregbool,'in_allorgs']),1/np.std(shortlisting.loc[noregbool,'deltaAIC']),1/np.std(maxelas)])
    weights = stdevs_inv/np.sum(stdevs_inv) # correct scores based on their standard deviation, so that all sources have same effect on the score
        
    for n in range(shortlisting.shape[0]): # calculate prioritization score
        shortlisting['score'].iloc[n] = (weights[0]*shortlisting['in_nmr'].iloc[n]+weights[1]*shortlisting['in_limpro'].iloc[n]+
                    weights[2]*shortlisting['in_allorgs'].iloc[n]+weights[3]*(-shortlisting['deltaAIC'].iloc[n])+
                    weights[4]*np.max(np.abs(shortlisting['elasticity_weighted'].iloc[n])))

    return shortlisting

#%% Show heatmap across conditions
# Take results and plot them as a heatmap.
# Inputs: results dataframe, reaction id.
    
def heatmap_across_conditions(results,rxn_id=None,save=False,save_dir=''):
    if rxn_id is not None: # if not specified, plot all reactions altogether
        results = results.loc[results['rxn_id']==rxn_id]
    cond = np.array(list(map(lambda x: x.shape[1],list(results['meas_flux'].values)))) # number of conditions
    cond_names = results.loc[cond==max(cond),'meas_flux'].iloc[0].columns.values # conditions names
    heat_mat = pd.DataFrame(columns=cond_names)
    for j,i in enumerate(list(results.index.values)): # add likelihood values in plotting matrix
        add = pd.DataFrame(results.loc[i,'lik_cond'].reshape([1,cond[j]]),columns=results.loc[i,'meas_flux'].columns,index=[i])
        heat_mat = heat_mat.append(add)
    fig, ax = plt.subplots()
    ax = heatmap(heat_mat,cmap='jet',xticklabels=cond_names) # plot matrix in heatmap
    ax.set_xlabel('Conditions')
    if rxn_id is not None:
        ax.set_yticks(np.arange(results.shape[0])+0.5)
        ax.set_yticklabels(results['regulator'].values,rotation = 0, ha="right")
        ax.set_title(str('%s: Fit likelihood across conditions' % rxn_id))
        if save:
            fig.savefig(save_dir+rxn_id+'_heat.pdf', bbox_inches='tight') # save figure
    else:
        ax.set_title('Fit likelihood across conditions')
        if save:
            fig.savefig(save_dir+'all_heatmap.pdf', bbox_inches='tight') # save figure
    plt.show()
    
#%% Plot predicted and measured fluxes
# Plot predicted and measured fluxes across conditions.
# Inputs: index or reaction id, results dataframe, summary dataframe, standard deviation of fluxes.
    
def plot_fit(idx,results,fluxes_sd=None,fullreg=None,save=False,save_dir=''):
    if isinstance(idx,int): # if idx is a number, plot only that model
        react = results.iloc[[idx]]
        width = 0.4
    elif isinstance(idx,str): # if idx is the name of a reaction, plot best models of that reaction
        react = results.loc[results['rxn_id']==idx][::-1]
        if len(react)>25: # plot top 25 models
            react = react.iloc[0:25,:]
            if '' not in react['regulator']: # plot the unregulated model if not in the top 25
                react = react.append(results.loc[(results['rxn_id']==idx)&(results['regulator']=='')])
        width = 0.8/(len(react)+1)
    meas_flux = react['meas_flux'] # observed fluxes
    pred_flux = react['pred_flux'].values # predicted fluxes
    sizes = list(map(lambda x:react['meas_flux'].iloc[x].shape[1],list(np.arange(len(react['meas_flux']))))) # number of conditions of each model
    ind = np.arange(max(sizes))
    fig, ax = plt.subplots()
    if meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].shape[0]==2: # if observed fluxes are min/max from FVA
        meas_flux_sd = (meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['max',:]-meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['min',:])/2 # standard deviation
        means = (meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['max',:]+meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['min',:])/2 # average
        plt.bar(ind, means.values.reshape(ind.shape), width, color='r', yerr=meas_flux_sd)
    elif fluxes_sd is None: # plot only point estimate if stdev is not available
        plt.bar(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), width, color='r')
    else: # plot point estimate and stdev
        meas_flux_sd = fluxes_sd.loc[react['rxn_id'].iloc[0],meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns]
        plt.bar(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), width, color='r', yerr=meas_flux_sd)
    if isinstance(idx,int): # single model
        plt.bar(ind + width, pred_flux[0][0].reshape(ind.shape), width, color='y') # plot predicted model
        plt.legend(['Measured', 'Predicted'])
        ax.set_title('%s%s: Flux fit between predicted and measured data' % (results['rxn_id'][idx],results['regulator'][idx]))
    elif isinstance(idx,str): # top models for a given reaction
        colors = cm.summer(np.arange(len(react))/len(react)) # set color scale
        for i in range(len(react)): # loop through all top models
            if (fullreg is not None) and (react['regulator'].iloc[i]!=''):
                if react['rxn_id'].iloc[i].lower() in list(fullreg.index.values):
                    cand = fullreg.loc[[react['rxn_id'].iloc[i].lower()],:] # known regulators for the reaction
                    for rg in react['regulator'].iloc[i]: # detect presence of model regulator in the known regulator list
                        reg = rg[4:-2]
                        if rg[0:3]=='ACT':
                            bools = [s=='+' for s in cand['mode']]; cand = cand.loc[bools,:]
                        else: 
                            bools = [s=='-' for s in cand['mode']]; cand = cand.loc[bools,:]
                        if reg in list(cand['metab']):
                            colors[i,:] = [1.0, 0.6, 0.0, 1.0] # show known regulators with a different color
            elif (react['regulator'].iloc[i]==''):
                colors[i,:] = [0.0, 0.6, 1.0, 1.0] # shown unregulated model with a different color
            if len(pred_flux[i][0]) < max(sizes): # if not all conditions are available for a certain model
                formated = np.array([0.0]*max(sizes))
                allcond = list(meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns)
                bools = np.array([False]*max(sizes))
                for j,cond in enumerate(allcond): # detect which conditions are missing
                    if any(cond in s for s in list(meas_flux.iloc[i].columns)):
                        bools[j]=True
                np.place(formated,bools,pred_flux[i][0]) # create array with 0s where conditions are not available
                plt.bar(ind + width*(1+i), formated.reshape(ind.shape), width, color = colors[i])
            else:
                plt.bar(ind + width*(1+i), pred_flux[i][0].reshape(ind.shape), width, color = colors[i])
        plt.legend(['Measured']+list(react['regulator'].values))
        ax.set_title('%s: Flux fit between predicted and measured data' % (react['rxn_id'].iloc[0]))
    ax.set_ylabel('Flux (mmol*gCDW-1*h-1)')
    ax.set_xticks(ind + 0.8 / 2 - width/2)
    ax.set_xticklabels(list(meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns),rotation = 30, ha="right")
    if save:
        fig.savefig(save_dir+'barflux_'+str(idx)+'.pdf', bbox_inches='tight')
    plt.show()
    
#%% Plot improvement of likelihood in best condition
# Plot likelihood improvement 
# Inputs: results dataframe, reaction id.
    
def plot_likelihood(results, cond=None, save=False, save_dir=''):
    noreg = results.loc[results['regulator']==''].reset_index(drop=True) # results of unregulated models
    leg_bool = True # legend boolean that determines which lagend is shown
    if isinstance(cond,str): # constrain plot to a single condition
        bool_cond = np.array(list(map(lambda x: any(cond in s for s in list(noreg['meas_flux'].iloc[x].columns)),list(np.arange(noreg.shape[0]))))) # reactions where the condition is available
        noreg = noreg.loc[bool_cond].reset_index(drop=True) # keep only models where condition is available
        bottom = np.array(list(map(lambda x: noreg['lik_cond'].iloc[x][cond==noreg['meas_flux'].iloc[x].columns][0],list(np.arange(noreg.shape[0]))))) # extract values of unregulated models
        top_1reg = np.array([0.0]*len(bottom))
        top_2reg = np.array([0.0]*len(bottom))
        for i,rxn in enumerate(list(noreg['rxn_id'].values)): # loop through reactions
            rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True) # regulated results of the reaction
            bool_rxn = np.array(list(map(lambda x: any(cond in s for s in list(rxn_results['meas_flux'].iloc[x].columns)),list(np.arange(rxn_results.shape[0]))))) # models where condition is available
            bool_1reg = np.array(list(map(lambda x: len(rxn_results['regulator'].iloc[x])==1,list(np.arange(rxn_results.shape[0]))))) # models with 1 regulator
            bool_2reg = np.array(list(map(lambda x: len(rxn_results['regulator'].iloc[x])>1,list(np.arange(rxn_results.shape[0]))))) # models with 2 or more regulators
            rxn_results_1reg = rxn_results[np.logical_and(bool_rxn,bool_1reg)].reset_index(drop=True) # extract results 1 reg
            rxn_results_2reg = rxn_results[np.logical_and(bool_rxn,bool_2reg)].reset_index(drop=True) # extract results >1 reg
            if (rxn_results_1reg.empty==False):
                lik_values_1reg = list(map(lambda x: rxn_results_1reg['lik_cond'].iloc[x][cond==rxn_results_1reg['meas_flux'].iloc[x].columns][0],list(np.arange(rxn_results_1reg.shape[0])))) # likelihood values 1 reg
                if bottom[i]<max(lik_values_1reg):
                    top_1reg[i] = max(lik_values_1reg)-bottom[i] # improvement of likelihood compared to unregulated model
            if (rxn_results_2reg.empty==False):
                lik_values_2reg = list(map(lambda x: rxn_results_2reg['lik_cond'].iloc[x][cond==rxn_results_2reg['meas_flux'].iloc[x].columns][0],list(np.arange(rxn_results_2reg.shape[0])))) # likelihood values >1 reg
                if (rxn_results_1reg.empty==False):
                    if (max(lik_values_1reg)<max(lik_values_2reg)):
                        top_2reg[i] = max(lik_values_2reg)-max(lik_values_1reg) # improvement of likelihood compared to 1 reg
                else:
                    if bottom[i]<max(lik_values_2reg):
                        top_2reg[i] = max(lik_values_2reg)-bottom[i] # improvement of likelihood compared to unregulated model
        xlabel = 'Reaction'
        title = str('Likelihood improvement in condition %s' % (cond))
        xticklabels = list(noreg['rxn_id'].values)
        ind = np.arange(len(bottom))+0.2
        fig, ax = plt.subplots()
        width = 0.4
        xticks = ind
        plt.bar(ind, bottom-min(bottom)+0.2, width, bottom=min(bottom)-0.2, color='r')
        plt.bar(ind, top_1reg, width, bottom=bottom, color='orange')
        if top_2reg.size!=0:
            leg_bool = False
            plt.bar(ind, top_2reg, width, bottom=top_1reg+bottom, color='gold')
    else: # plot all conditions altogether
        min_value = np.nanmin(np.concatenate(results['lik_cond'].values)) # find minimum likelihood across all models and conditions
        ncond = np.array(list(map(lambda x: noreg['meas_flux'].iloc[x].shape[1],list(np.arange(noreg.shape[0]))))) # number of conditions of unregulated models
        conds = list(noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns) # detect minimum set of common conditions across all recations
        ind = np.arange(noreg.shape[0])
        top_1reg = pd.DataFrame(data=np.zeros((noreg.shape[0],max(ncond))),index=noreg['rxn_id'],columns=noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        top_2reg = pd.DataFrame(data=np.zeros((noreg.shape[0],max(ncond))),index=noreg['rxn_id'],columns=noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        bottom = pd.DataFrame(index=noreg['rxn_id'],columns=noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        fig, ax = plt.subplots()
        title = 'Likelihood improvement'
        width = 0.8/len(conds)
        xticks = ind + 0.8 / 2 - width/2
        for j,cond in enumerate(conds): # loop through all common conditons
            bool_cond = np.array(list(map(lambda x: any(cond in s for s in list(noreg['meas_flux'].iloc[x].columns)),list(np.arange(noreg.shape[0]))))) # reactions where the condition is available
            noreg2 = noreg.loc[bool_cond].reset_index(drop=True) # keep only models where condition is available
            bottom.loc[bool_cond,cond] = np.array(list(map(lambda x: noreg2['lik_cond'].iloc[x][cond==noreg2['meas_flux'].iloc[x].columns][0],list(np.arange(noreg2.shape[0]))))) # extract values of unregulated models
            for i,rxn in enumerate(list(noreg2['rxn_id'].values)): # loop through reactions
                rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True) # regulated results of the reaction
                bool_rxn = np.array(list(map(lambda x: any(cond in s for s in list(rxn_results['meas_flux'].iloc[x].columns)),list(np.arange(rxn_results.shape[0]))))) # models where condition is available
                bool_1reg = np.array(list(map(lambda x: len(rxn_results['regulator'].iloc[x])==1,list(np.arange(rxn_results.shape[0]))))) # models with 1 regulator
                bool_2reg = np.array(list(map(lambda x: len(rxn_results['regulator'].iloc[x])>1,list(np.arange(rxn_results.shape[0]))))) # models with 2 or more regulators
                rxn_results_1reg = rxn_results.loc[np.logical_and(bool_rxn,bool_1reg)].reset_index(drop=True) # extract results 1 reg
                rxn_results_2reg = rxn_results.loc[np.logical_and(bool_rxn,bool_2reg)].reset_index(drop=True) # extract results >1 reg
                if (rxn_results_1reg.empty==False):
                    lik_values_1reg = list(map(lambda x: rxn_results_1reg['lik_cond'].iloc[x][cond==rxn_results_1reg['meas_flux'].iloc[x].columns][0],list(np.arange(rxn_results_1reg.shape[0])))) # likelihood values 1 reg
                    if bottom.loc[rxn,cond]<max(lik_values_1reg):
                        top_1reg.loc[rxn,cond] = max(lik_values_1reg)-bottom.loc[rxn,cond] # improvement of likelihood compared to unregulated model
                if (rxn_results_2reg.empty==False):
                    lik_values_2reg = list(map(lambda x: rxn_results_2reg['lik_cond'].iloc[x][cond==rxn_results_2reg['meas_flux'].iloc[x].columns][0],list(np.arange(rxn_results_2reg.shape[0])))) # likelihood values >1 reg
                    if (rxn_results_1reg.empty==False):
                        if (max(lik_values_1reg)<max(lik_values_2reg)):
                            top_2reg.loc[rxn,cond] = max(lik_values_2reg)-max(lik_values_1reg) # improvement of likelihood compared to 1 reg
                    else:
                        if bottom.loc[rxn,cond]<max(lik_values_2reg):
                            top_2reg.loc[rxn,cond] = max(lik_values_2reg)-bottom.loc[rxn,cond] # improvement of likelihood compared to unregulated model
            plt.bar(ind+width*(j), bottom[cond].values-min_value+0.2, width, bottom=min_value-0.2, color='r')
            plt.bar(ind+width*(j), top_1reg[cond].values, width, bottom=bottom[cond].values, color='orange')
            if any(np.isnan(list(top_2reg.loc[:,cond]))==0):
                leg_bool = False # if any >1 reg result showing up, change legend
                plt.bar(ind+width*(j), top_2reg[cond].values, width, bottom=top_1reg[cond].values+bottom[cond].values, color='gold')
        
    xlabel = 'Reaction'
    xticklabels = list(noreg['rxn_id'].values)
    ax.set_ylabel('Likelihood')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,rotation = 45, ha="right")
    if leg_bool: # set correct legend
        plt.legend(['General M-M','1 Regulator'],loc='upper left')
    else:
        plt.legend(['General M-M','1 Regulator','>1 Regulator'],loc='upper left')
    if save:
        fig.savefig(save_dir+'improvement_'+str(cond)+'.pdf', bbox_inches='tight') # save figure
    plt.show()

#%% Plot correlation prediction vs measured flux
# Plot correlation between predicted and measured fluxes across conditions.
# Inputs: results dataframe.
    
def plot_corr(results,xlabel='rxn_id',save=False,save_dir=''):
    yplot = [(pearsonr(results.loc[i,'meas_flux'].values,results.loc[i,'pred_flux'][0])[0])**2 if results.loc[i,'meas_flux'].shape[0]==1 else (pearsonr(np.nanmean(results.loc[i,'meas_flux'].values,0),results.loc[i,'pred_flux'][0])[0])**2 for i in results.index] # calculate Pearson determination coefficient
    xlabels = list(results.loc[np.isnan(yplot)!=True,xlabel]) # extract x axis labels
    yplot = np.array(yplot)[np.isnan(yplot)!=True] # remove reactions with NaN
    to_return = pd.DataFrame(yplot, index=xlabels)
    xlabels = sorted(xlabels,key=lambda i: float(yplot[[i in s for s in xlabels]])) # order reactions from low to high R2
    yplot = np.array(sorted(yplot))
    xplot = np.arange(len(yplot))
    plt.scatter(xplot[yplot>=.35],yplot[yplot>=.35],c='g') # green for high R2
    plt.scatter(xplot[yplot<.35],yplot[yplot<.35],c='r') # red for low R2
    plt.legend(['R2 >= 0.35', 'R2 < 0.35'])
    plt.title('Fit of Michaelis-Menten prediction')
    plt.ylabel('Pearson Determination Coefficient')
    plt.xlabel(xlabel)
    plt.xticks(xplot,xlabels,rotation = 50, ha="right")
    if save:
        plt.savefig(save_dir+'correlation.pdf', bbox_inches='tight')
    plt.show()
    return to_return

#%% Plot predicted and measured fluxes
# Plot predicted and measured fluxes across conditions.
# Inputs: index or reaction id, results dataframe, summary dataframe, standard deviation of fluxes.
    
def plot_scatter(idx,results,edge=None,fluxes_sd=None,save=False,save_dir=''):
    react = results.loc[[idx]] # reaction results
    meas_flux = react['meas_flux'] # observed fluxes
    pred_flux = react['pred_flux'].values # predicted fluxes
    noreg_flux = results.loc[np.logical_and(results['rxn_id']==str(react['rxn_id'].values[0]),results['regulator']==''),'pred_flux'].values # predicted fluxes of unregulated model
    sizes = list(map(lambda x:react['meas_flux'].iloc[x].shape[1],list(np.arange(len(react['meas_flux']))))) # number of available conditions
    ind = np.arange(max(sizes))
    fig, ax = plt.subplots()
    if meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].shape[0]==2: # if flux is min/max from FVA
        meas_flux_sd = (meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['max',:]-meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['min',:])/2 # standard deviation
        means = (meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['max',:]+meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].loc['min',:])/2 # average
        ax.scatter(ind, means.values.reshape(ind.shape),color='r') # plot points
        ax.errorbar(ind, means.values.reshape(ind.shape),yerr=meas_flux_sd,fmt='none')
        if edge is None: # plot line between points
            meas, = ax.plot(ind, means.values.reshape(ind.shape), color='r',lw=2.0)
        else:
            start=0
            for i in edge: # edge specifies groups of conditions. Ex: [4,5], for 4 glucose and 5 glutamate limitations
                meas, = ax.plot(ind[start:start+i], means.values.reshape(ind.shape)[start:start+i], color='r',lw=2.0)
                start += i
    elif fluxes_sd is None: # plot point estimate of fluxes
        ax.scatter(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape),color='r') # plot points
        if edge is None: # plot line between points
            meas, = ax.plot(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), color='r',lw=2.0)
        else:
            start=0
            for i in edge: # edge specifies groups of conditions. Ex: [4,5], for 4 glucose and 5 glutamate limitations
                meas, = ax.plot(ind[start:start+i], meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape)[start:start+i], color='r',lw=2.0)
                start += i
    else: # plot point estimate of fluxes and stdev
        meas_flux_sd = fluxes_sd.loc[react['rxn_id'].iloc[0],meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns]
        ax.scatter(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), color='r') # plot points
        ax.errorbar(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape),yerr=meas_flux_sd,fmt='none')
        if edge is None: # plot line between points
            meas, = ax.plot(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), color='r',lw=2.0)
        else:
            start=0
            for i in edge: # edge specifies groups of conditions. Ex: [4,5], for 4 glucose and 5 glutamate limitations
                meas, = ax.plot(ind[start:start+i], meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape)[start:start+i], color='r',lw=2.0)
                start += i
    ax.scatter(ind, pred_flux[0][0].reshape(ind.shape), color='b') # plot points
    ax.scatter(ind, noreg_flux[0][0].reshape(ind.shape), color='b') # plot points
    if edge is None:  # plot line between points
        pred, = ax.plot(ind, pred_flux[0][0].reshape(ind.shape), color='b',lw=2.0)
        noreg, = ax.plot(ind, noreg_flux[0][0].reshape(ind.shape), color='b',lw=2.0,ls=':')
    else: # edge specifies groups of conditions. Ex: [4,5], for 4 glucose and 5 glutamate limitations
        start=0
        for i in edge:
            pred, = ax.plot(ind[start:start+i], pred_flux[0][0].reshape(ind.shape)[start:start+i], color='b',lw=2.0)
            noreg, = ax.plot(ind[start:start+i], noreg_flux[0][0].reshape(ind.shape)[start:start+i], color='b',lw=2.0,ls=':')
            start += i
    ax.legend([meas,pred,noreg],['Measured', 'Regulated Model','Base Model'])
    ax.set_title('%s%s: Flux fit between predicted and measured data' % (results['rxn_id'][idx],results['regulator'][idx]))
    ax.set_ylabel('Flux (mmol*gCDW-1*h-1)')
    ax.set_xticks(ind)
    ax.set_xticklabels(list(meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns),rotation = 30, ha="right")
    if save:
        fig.savefig(save_dir+'scatter_'+str(idx)+'.pdf', bbox_inches='tight')
    plt.show()