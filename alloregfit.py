# -*- coding: utf-8 -*-

#%% Import modules
import pandas as pd
import sympy as sym
import numpy as np
from obonet import read_obo
from numpy.random import uniform,normal
from scipy.stats import norm
from scipy.optimize import nnls
from scipy.integrate import quad
from math import pi,e
import matplotlib.pyplot as plt
from seaborn import heatmap
from matplotlib import cm

#%% Extract available molecules
# For all the molecules involved in the reaction, extract the corresponding omics data.
# Inputs: molecules to extract info, omics dataframe, and type of molecule (reactant,
# product or enzyme).
def extract_info_df(molecules, dataset, mol_type):
    mol_df = []
    names = []
    ncond = dataset.shape[1]
    mol_bool = [True]*ncond
    if mol_type=='enzyme':
        mapping = pd.read_table("ECOLI_83333_idmapping.dat",header=None)
        for j in range(len(molecules)):
            gene = mapping[mapping[2]==molecules[j].id][0].reset_index()
            gene = list(mapping[((mapping[0]==gene[0][0]) & (mapping[1]=='Gene_Name'))][2])
            if any(gene[0] in s for s in [dataset.index.values]):
                mol_df.append(dataset.loc[gene[0]].values)
                names.append(gene[0])
                mol_bool = (mol_bool & (np.isnan(dataset.loc[gene[0]].values)==0))
    else:
        for j in range(len(molecules)):
            met = molecules[j].id[:-2] #strip compartment letter from id
            if ((mol_type=='reactant')and(any(met in s for s in ['h','h2o'])==0)) or \
            ((mol_type=='product')and(any(met in s for s in [dataset.index.values]) and \
              (any(met in s for s in ['h','h2o'])==0))):
                    mol_df.append(dataset.loc[met].values)
                    names.append(molecules[j].id)
                    mol_bool = (mol_bool & (np.isnan(dataset.loc[met].values)==0))
    mol_df = pd.DataFrame(mol_df,columns = dataset.columns, index = names)
    return mol_df, mol_bool
    
#%% Determine binding site
# For all metabolites involved in reaction, determine the binding site.
# Inputs: reactants, products, model.
    
def get_binding_sites(rxn_id,model):
    obo_map = read_obo("chebi_lite.obo")
    bigg_chebi = pd.read_csv("bigg_to_chebi2.csv", index_col = 'bigg_id')
    binding_site = []
    for i in range(len(rxn_id)):
        all_met = [x.id for x in model.reactions.get_by_id(rxn_id[i]).metabolites.keys()]
        all_met = [s for s in all_met if s not in ['h_c','h2o_c','pi_c']]
        bs_array = np.zeros([len(all_met),len(all_met)])
        all_chebi = list(map(lambda x: int(bigg_chebi.loc[x]),all_met))
        id_to_altid = {id_:data['alt_id'] for id_, data in obo_map.nodes(data=True) if any('alt_id' in s for s in list(data.keys()))}
        altid_to_id = {}
        for k,v in id_to_altid.items():
            for alt in v:
                altid_to_id[alt]=k
        ori_id = [altid_to_id[str('CHEBI:%i' % ori)] if any(str('CHEBI:%i' % ori) in s for s in list(altid_to_id.keys())) else str('CHEBI:%i'%ori) for ori in all_chebi]
        for r,met1 in enumerate(ori_id):
            neigh1 = obo_map.successors(met1)
            for j,met2 in enumerate(ori_id):
                neigh2 = obo_map.successors(met2)
                if any([True for n1, n2 in zip(neigh1, neigh2) if n1 == n2]) and r<=j:
                    bs_array[r,j] = 1
        bind_site = []
        all_met = np.array(all_met)
        while (all_met.shape[0])>1:
            bind_site.append(list(all_met[bs_array[0]==1]))
            all_met = all_met[bs_array[0,:]==0]
            bs_array = bs_array[1:,bs_array[0,:]==0]
        else:
            if (all_met.shape[0])>0:
                bind_site.append([str(all_met[0])])
            
        print(model.reactions.get_by_id(rxn_id[i]).reaction)
        print(bind_site)
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

def define_reactions(rxn_id, model, fluxes, prot, metab, binding_site=None):
    reaction, reactant, product, enzyme, flux, bools, bs = ([] for l in range(7))
    
    for i in range(len(rxn_id)):
        # Reaction value
        reaction.append(model.reactions.get_by_id(rxn_id[i]).reaction)
        # Reactant values
        react = model.reactions.get_by_id(rxn_id[i]).reactants
        react_df, react_bool = extract_info_df(react,metab,'reactant')
        # Product values
        prod = model.reactions.get_by_id(rxn_id[i]).products
        prod_df, prod_bool = extract_info_df(prod,metab,'product')
        # Enzyme values
        enz = list(model.reactions.get_by_id(rxn_id[i]).genes)
        enz_df, enz_bool = extract_info_df(enz,prot,'enzyme')
        # Append all data
        flux_bool = np.isnan(fluxes.loc[rxn_id[i]].values)==0
        bool_all = (react_bool & prod_bool & enz_bool & flux_bool)
        reactant.append(react_df.loc[:,bool_all])
        product.append(prod_df.loc[:,bool_all])
        enzyme.append(enz_df.loc[:,bool_all])
        flux.append(pd.DataFrame([fluxes.loc[rxn_id[i]].values],columns = fluxes.columns, index = [rxn_id[i]]).loc[:,bool_all])
        bools.append(bool_all)
        if binding_site is None:
            bs.append([list(react_df.index.values)+list(prod_df.index.values)])
    
    if binding_site is None:
        binding_site=bs
        
    summary = pd.DataFrame({'idx':range(len(rxn_id)),'reaction':reaction,'rxn_id':rxn_id,\
                            'reactant':reactant,'product':product,\
                            'enzyme':enzyme,'flux':flux,'binding_site':binding_site})
    summary = summary.set_index('idx')
    return summary,bools

#%% Define candidates
# For each reaction, a table with the regulators is created. 
# Inputs: list of reaction ids that will be analyzed, DataFrame with regulators
# for all rxn_id in E. coli, DataFrame with metabolites x cond, and DataFrame with
# regulators for all rxn_id in other organisms (optional).
    
def define_candidates(rxn_id,reg_coli,metab,bools,reg_other=None):
    act_coli, inh_coli, act_other, inh_other = ([] for l in range(4))
    for i in range(len(rxn_id)):
        if (any(rxn_id[i].lower() in s for s in [reg_coli.index.values])):
            cand_coli = reg_coli.loc[[rxn_id[i].lower()]].reset_index()
            act_coli_df,name_act_coli,inh_coli_df,name_inh_coli = ([] for l in range(4))
            for j,met in enumerate(list(cand_coli['metab'])):
                if (any(met in s for s in [metab.index.values])):
                    if cand_coli['mode'][j] == '-':
                        inh_coli_df.append(metab.loc[met].values)
                        name_inh_coli.append(met+'_c')
                    elif cand_coli['mode'][j] == '+':
                        act_coli_df.append(metab.loc[met].values)
                        name_act_coli.append(met+'_c')
            inh_coli_df = pd.DataFrame(inh_coli_df,columns = metab.columns, index=name_inh_coli)
            act_coli_df = pd.DataFrame(act_coli_df,columns = metab.columns, index=name_act_coli)
            if act_coli_df.empty:
                act_coli.append('No data available for the candidate activators.')
            else:
                act_coli_df.drop_duplicates(inplace=True); act_coli.append(act_coli_df.loc[:,bools[i]])
            if inh_coli_df.empty:
                inh_coli.append('No data available for the candidate activators.')
            else:
                inh_coli_df.drop_duplicates(inplace=True); inh_coli.append(inh_coli_df.loc[:,bools[i]])
        else:
            act_coli.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
            inh_coli.append('No candidate regulators for %s in E.coli.' % rxn_id[i])
        if reg_other is None:
            act_other.append([None]*len(rxn_id))
            inh_other.append([None]*len(rxn_id))
        else:
            pass
    candidates = pd.DataFrame({'idx':range(len(rxn_id)),'act_coli':act_coli,'inh_coli':inh_coli,\
                            'act_other':act_other,'inh_other':inh_other})
    candidates = candidates.set_index('idx')
    return candidates        

#%% Write regulator expression
# For each regulator, write the regulatory expression to add.
# Inputs: list of regulators and their +/- effect.
    
def write_reg_expr(regulators,reg_type,coop=False):
    add, newframe, reglist = ([] for l in range(3))
    for reg in regulators:
        R = str('c_%s' % reg)
        K = str('K_%s' % reg)
        if coop is False:
            if reg_type=='activator':
                add.append(sym.sympify(R+'/('+R+'+'+K+')'))
                reglist.append('ACT:'+reg)
            elif reg_type=='inhibitor':
                add.append(sym.sympify('1/(1+('+R+'/'+K+'))'))
                reglist.append('INH:'+reg)
            new_par = [K]; new_spe = [reg]; new_spetype = ['met']
        elif coop is True:
            n = str('n_%s' % reg)
            if reg_type=='activator':
                add.append(sym.sympify(R+'**'+n+'/('+R+'**'+n+'+'+K+'**'+n+')'))
                reglist.append('ACT:'+reg)
            elif reg_type=='inhibitor':
                add.append(sym.sympify('1/(1+('+R+'/'+K+')**'+n+')'))
                reglist.append('INH:'+reg)
            new_par = [K,n]; new_spe = [reg,'hill']; new_spetype = ['met','hill']
        newframe.append(pd.DataFrame({'parameters':new_par,'species':new_spe,'speciestype':new_spetype}))
    return add, newframe, reglist

#%% Add regulators
# For all kind of regulators, generate a structure containing all expressions to add, and their respective parameters.
# Inputs: candidates dataframe
def add_regulators(idx,candidates,coop=False):
    add, newframe, reg = ([] for l in range(3))
    if (list(candidates.columns.values)==['act_coli', 'act_other', 'inh_coli', 'inh_other']):
        if isinstance(candidates['act_coli'][idx],pd.DataFrame):
            act_coli = list(candidates['act_coli'][idx].index)
            add1, newframe1, reg1 = write_reg_expr(act_coli,'activator',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['inh_coli'][idx],pd.DataFrame):
            inh_coli = list(candidates['inh_coli'][idx].index)
            add1, newframe1, reg1 = write_reg_expr(inh_coli,'inhibitor',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['act_other'][idx],pd.DataFrame):
            act_other = list(candidates['act_other'][idx].index)
            add1, newframe1, reg1 = write_reg_expr(act_other,'activator',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
        if isinstance(candidates['inh_other'][idx],pd.DataFrame):
            inh_other = list(candidates['inh_other'][idx].index)
            add1, newframe1, reg1 = write_reg_expr(inh_other,'inhibitor',coop)
            add.extend(add1); newframe.extend(newframe1); reg.extend(reg1)
    else:
        cand = list(map(lambda x: x+'_c',list(candidates.index)))
        add1, newframe1, reg1 = write_reg_expr(cand,'activator',coop)
        add2, newframe2, reg2 = write_reg_expr(cand,'inhibitor',coop)
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
    for enz in enzyme:
        K = str('K_cat_%s' % enz)
        E = str('c_%s' % enz)
        vmax += sym.sympify(K+'*'+E)
        
    # Define occupancy term. Start with the numerator:
    reaction = model.reactions.get_by_id(summary['rxn_id'][idx])
    substrate = list(summary['reactant'][idx].index)
    num1 = sym.sympify('1')
    num2 = sym.sympify('1')
    for sub in substrate:
        K = str('K_%s' % sub)
        num1 *= sym.sympify(K)
        S = str('c_%s' % sub)
        exp = abs(reaction.get_coefficient(sub))
        num2 *= sym.sympify(S+'**'+str(exp))
        parameters.append(K), species.append(sub), speciestype.append('met')
    num1 = 1/num1            
    
    product = list(summary['product'][idx].index)
    if product:
        num3 = sym.sympify('1')
        for prod in product:
            P = str('c_%s' % prod)
            exp = abs(reaction.get_coefficient(prod))
            num3 *= sym.sympify(P+'**'+str(exp))
        K_eq = sym.symbols('K_eq')
        parameters.append('K_eq'), species.append('K_eq'), speciestype.append('K_eq')
        num3 = (1/K_eq)*num3
        num = num1*(num2-num3)
    else:
        num = num1*num2
    
    # Define the denominator:
    den = sym.sympify('1')
    for i,site in enumerate(summary['binding_site'][idx]):
        den_site = sym.sympify('1')
        for met in summary['binding_site'][idx][i]:
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
    parframe[0].drop_duplicates('parameters',inplace=True)
    parframe[0].reset_index(drop=True,inplace=True)
    regulator = ['']
    
    if (candidates is not None) and (nreg>=1):
        add, newframe, reg = add_regulators(idx,candidates,coop)
        for i in range(len(add)):
            expr.append({'vmax':vmax,'occu':add[i]*(num/den)})
            addframe = parframe[0].append(newframe[i])
            addframe.drop_duplicates('parameters',inplace=True)
            addframe.reset_index(drop=True,inplace=True)
            parframe.append(addframe)
            regulator.append([reg[i]])
            if nreg>=2:
                for j in range(len(add)):
                    if i>j:
                        expr.append({'vmax':vmax,'occu':add[j]*add[i]*(num/den)})
                        addframe = parframe[0].append(newframe[i])
                        addframe = addframe.append(newframe[j])
                        addframe.drop_duplicates('parameters',inplace=True)
                        addframe.reset_index(drop=True,inplace=True)
                        parframe.append(addframe)
                        regulator.append([reg[i],reg[j]])
    
    return expr,parframe,regulator

#%% Build parameter priors
# For each of the parameters, define the prior/proposal distribution needed for MCMC.
# Inputs: dataframe with parameters, summary generated in define_reactions, the 
# stoichiometric model, and candidate dataframe.            
def build_priors(param, idx, summary, model, priorKeq=False, candidates=None):
    reaction = model.reactions.get_by_id(summary['rxn_id'][idx])
    distribution, par1, par2 = ([] for i in range(3))
    for i,par in enumerate(param['parameters']):
        if param['speciestype'][i] == 'met':
            distribution.append('unif')
            if any(param['species'][i] in s for s in [summary['reactant'][idx].index.values]):
                par1.append(-15.0+np.log2(np.nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+np.log2(np.nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
            elif any(param['species'][i] in s for s in [summary['product'][idx].index.values]):
                par1.append(-15.0+np.log2(np.nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+np.log2(np.nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
            elif (candidates is not None):
                if list(candidates.columns.values)==['act_coli', 'act_other', 'inh_coli', 'inh_other']:
                    if (isinstance(candidates['act_coli'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['act_coli'][idx].index.values])):
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['act_coli'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['act_coli'][idx].loc[param['species'][i]].values)))
                    elif (isinstance(candidates['inh_coli'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['inh_coli'][idx].index.values])):
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['inh_coli'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['inh_coli'][idx].loc[param['species'][i]].values)))
                    elif (isinstance(candidates['act_other'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['act_other'][idx].index.values])):
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['act_other'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['act_other'][idx].loc[param['species'][i]].values)))
                    elif (isinstance(candidates['inh_other'][idx],pd.DataFrame)) and \
                    (any(param['species'][i] in s for s in [candidates['inh_other'][idx].index.values])):
                        par1.append(-15.0+np.log2(np.nanmedian(candidates['inh_other'][idx].loc[param['species'][i]].values)))
                        par2.append(15.0+np.log2(np.nanmedian(candidates['inh_other'][idx].loc[param['species'][i]].values)))
                else:
                    cond = summary['flux'][idx].columns.values
                    par1.append(-15.0+np.log2(np.nanmedian(candidates.loc[param['species'][i][:-2],cond].values)))
                    par2.append(15.0+np.log2(np.nanmedian(candidates.loc[param['species'][i][:-2],cond].values)))
        elif param['speciestype'][i] == 'K_eq':
            Q_r = 1
            for subs in list(summary['reactant'][idx].index):
                Q_r /= (summary['reactant'][idx].loc[subs].values)**abs(reaction.get_coefficient(subs))
            products = list(summary['product'][idx].index.values)
            if products:
                for prod in products:
                    Q_r *= (summary['product'][idx].loc[prod].values)**abs(reaction.get_coefficient(prod))
                if priorKeq==False:
                    distribution.append('unif')
                    par1.append(-20.0+np.log2(np.nanmedian(Q_r)))
                    par2.append(20.0+np.log2(np.nanmedian(Q_r)))
                else:
                    priorKeqs = pd.read_csv('Keq_bigg.csv',index_col='bigg_id')
                    bools = np.array(list(isinstance(x,str) for x in list(priorKeqs.index.values)))
                    if any(summary['rxn_id'][idx] in s for s in list(priorKeqs.loc[bools].index)):
                        distribution.append('norm')
                        par1.append(np.log2(float(priorKeqs.loc[summary['rxn_id'][idx],'Keq'])))
                        par2.append(np.abs(np.log2(float(priorKeqs.loc[summary['rxn_id'][idx],'Keq']))-np.log2(np.nanmedian(Q_r))))
                    else:
                        distribution.append('unif')
                        par1.append(-20.0+np.log2(np.nanmedian(Q_r)))
                        par2.append(20.0+np.log2(np.nanmedian(Q_r)))
        elif param['speciestype'][i] == 'hill':
            distribution.append('unif')
            par1.append(-3)
            par2.append(3)
    param['distribution'] = pd.Series(distribution, index=param.index)
    param['par1'] = pd.Series(par1, index=param.index)
    param['par2'] = pd.Series(par2, index=param.index)
    return param

#%% Draw parameters
# From the prior distribution, update those parameters that are present in ‘updates’.
# Inputs: parameter indeces to update within a list, parameter dataframe with priors, current values.
def draw_par(update, parameters, current):
    draw = current
    for par in update:
        if parameters['distribution'][par]=='unif':
            draw[par] = uniform(parameters['par1'][par],parameters['par2'][par])
        elif parameters['distribution'][par]=='norm':
            draw[par] = normal(parameters['par1'][par],parameters['par2'][par])
        else:
            print('Invalid distribution')
    return draw


#%% Calculate likelihood between flux range
# Given the flux prediction and min/max fluxes from FVA, integrate normal PDF calculation over this range.
# Inputs: min/max flux and prediction.
def range_PDF(pred_flux, flux, ncond, npars):
    def PDF(x,y):
        return (1/(np.sqrt(np.sum((x-y)**2)/(ncond-npars))*np.sqrt(2*pi)))*e**(-(x-y)**2/(2*(np.sum((x-y)**2)/(ncond-npars))))
    likelihood = np.array(list(map(lambda i: quad(PDF,flux[0][i],flux[1][i],args=pred_flux[0][i]),\
                                   [n for n in range(pred_flux.size)])))
    return likelihood[:,0]
    
#%% Calculate likelihood
# Calculate the likelihood given the flux point estimate or the lower and upper bounds of flux variability analysis.
# Inputs: parameter dataframe, current values, summary as generated in define_reactions, 
# equations, and candidates.
def calculate_lik(idx,parameters, current, summary, equations,candidates=None,regulator=None):
    occu = equations['occu']
    enz = summary['enzyme'][idx].values
    ncond = enz.shape[1]
    current = np.array(current)
    vbles = []
    vbles_vals = []
    for par in list(parameters['parameters'].values):
        vbles.append(par)
        rep_par = np.repeat(2**current[parameters['parameters'].values==par],ncond)
        vbles_vals.append(rep_par)
    for sub in list(summary['reactant'][idx].index):
        vbles.append('c_'+sub)
        vbles_vals.append(summary['reactant'][idx].loc[sub].values)
    for prod in list(summary['product'][idx].index):
        vbles.append('c_'+prod)
        vbles_vals.append(summary['product'][idx].loc[prod].values)
    if regulator:
        for reg in regulator:
            reg = reg[4:]
            if list(candidates.columns.values)==['act_coli', 'act_other', 'inh_coli', 'inh_other']:
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
            elif not (any('c_'+reg in s for s in vbles)):
                cond = summary['flux'][idx].columns.values
                vbles.append('c_'+reg)
                vbles_vals.append(candidates.loc[reg[:-2],cond].values)
                
    f = sym.lambdify(vbles, occu)
    bool_all =(np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0))==max(np.sum(np.dot(list(map(lambda x: (np.isnan(x)==0),vbles_vals)),1),0))
    vbles_vals = list(map(lambda x: x[bool_all],vbles_vals))
    ncond = np.sum(bool_all)
    pred_occu = f(*vbles_vals)
    flux = summary['flux'][idx].values
    if len(summary['flux'][idx]) == 1: # fluxes as point estimates
        kcat, residual = nnls(np.transpose(pred_occu*enz[:,bool_all]), flux[:,bool_all].reshape(ncond))
        pred_flux = np.sum(kcat*np.transpose(enz[:,bool_all]),1)*pred_occu
        npars = len(current)+len(kcat)
        if ncond>npars:
            var = np.sum((flux[:,bool_all]-pred_flux)**2)/(ncond-npars)
            likelihood = norm.pdf(flux[:,bool_all], pred_flux, np.sqrt(var))
            return np.sum(np.log(likelihood)),kcat,pred_flux,np.log(likelihood),bool_all
        else:
            return None,None,None,None,None # Quit the evaluation of this expression
    elif len(summary['flux'][idx]) == 2: # min/max range of fluxes
        ave_flux = np.mean(flux,0)
        kcat, residual = nnls(np.transpose(pred_occu*enz[:,bool_all]), ave_flux[:,bool_all])
        pred_flux = np.sum(kcat*np.transpose(enz[:,bool_all]),1)*pred_occu
        npars = len(current)+len(kcat)
        if ncond>npars:
            likelihood = (1/(flux[0,bool_all]-flux[1,bool_all]))*range_PDF(pred_flux, flux[:,bool_all], ncond, npars)
            return np.sum(np.log(likelihood)),kcat,pred_flux,np.log(likelihood),bool_all
        else:
            return None,None,None,None,None # Quit the evaluation of this expression

#%% Fit reaction equation using MCMC-NNLS
# Sample posterior distribution Pr(Ω|M,E,jF) using MCMC-NNLS.
# Inputs: markov parameters (fraction of samples that are reported, how many samples are 
# desired, how many initial samples are skipped), parameters table with priors, 
# summary as generated in define_reactions, equations, and candidates.
def fit_reaction_MCMC(idx, markov_par, parameters, summary, equations,candidates=None,regulator=None):
    print('Running MCMC-NNLS for reaction %d... Candidate regulator: %s' % (idx,regulator))
    colnames = list(parameters['parameters'].values)
    colnames.extend(['kcat','likelihood','pred_flux','lik_cond'])
    track = pd.DataFrame(columns=colnames)
    current_pars = [None] * len(parameters)
    current_pars = draw_par([p for p in range(len(parameters))], parameters, current_pars)
    current_lik,cur_kcat,cur_pred_flux,cur_lik_cond,bool_all = calculate_lik(idx, parameters, current_pars, summary, equations,candidates,regulator)
    if current_lik is None:
        print('Number of parameters outpaces the number of conditions.')
        return None,None # Quit the evaliation of this expression
    else:
        for i in range(markov_par['burn_in']+markov_par['nrecord']*markov_par['freq']):
            for p in range(len(parameters)):
                proposed_pars = draw_par([p], parameters, current_pars)
                proposed_lik,pro_kcat,pro_pred_flux,pro_lik_cond,bool_all = calculate_lik(idx, parameters, proposed_pars, summary, equations,candidates,regulator)
                if ((uniform(0,1) < np.exp(proposed_lik)/(np.exp(proposed_lik)+np.exp(current_lik))) or \
                    (proposed_lik > current_lik) or ((proposed_lik==current_lik)and(proposed_lik==-np.inf))):
                    current_pars = proposed_pars
                    cur_kcat = pro_kcat
                    cur_pred_flux = pro_pred_flux
                    current_lik = proposed_lik
                    cur_lik_cond = pro_lik_cond
            if (i > markov_par['burn_in']):
                if ((i-markov_par['burn_in'])%markov_par['freq'])==0:
                    add_pars = list(map(lambda x: 2**x, current_pars))
                    add_pars.append(cur_kcat); add_pars.append(current_lik); add_pars.append(cur_pred_flux); add_pars.append(cur_lik_cond)
                    track = track.append(pd.DataFrame([add_pars],columns=colnames))
        track.reset_index(drop=True,inplace=True)
        return track,bool_all

#%% Fit reaction equations
# Run all required functions as a block to fit predicted to measured flux.
# Inputs: summary dataframe, stoichiometric model, markov parameters, and candidates dataframe (optional).

def fit_reactions(summary,model,markov_par,priorKeq=False,candidates=None,maxreg=1,coop=False):
    results = pd.DataFrame(columns=['idx','reaction','rxn_id','regulator','equation',\
                                    'meas_flux','pred_flux','best_fit','best_lik','lik_cond'])
    for idx in list(summary.index):
        expr,parameters,regulator = write_rate_equations(idx,summary,model,candidates,maxreg,coop)
        for i in range(len(expr)):
            parameters[i] = build_priors(parameters[i],idx,summary,model,priorKeq,candidates)
            track,bool_all = fit_reaction_MCMC(idx,markov_par,parameters[i],summary,expr[i],candidates,regulator[i])
            if track is None:
                continue # Quit the evaliation of this expression
            else:
                max_lik = max(track['likelihood'].values)
                max_par = track[track['likelihood'].values==max_lik]
                add = {'idx':idx,'reaction':summary['reaction'][idx],'rxn_id':summary['rxn_id'][idx],\
                       'regulator':regulator[i],'equation':sym.latex(expr[i]['vmax']*expr[i]['occu']),'meas_flux':summary['flux'][idx].loc[:,bool_all],\
                       'pred_flux':max_par.iloc[:,-2].values,'best_fit':max_par.iloc[:,:-3],'best_lik':max_lik,\
                       'lik_cond':max_par.iloc[:,-1].values[0]}
                results = results.append([add])
    results = results.sort_values(by='best_lik')
    results.reset_index(drop=True,inplace=True)
    return results

#%% Show heatmap across conditions
# Take results and plot them as a heatmap.
# Inputs: results dataframe, reaction id.

def heatmap_across_conditions(results,rxn_id=None,save=False,save_dir=''):
    if rxn_id is not None:
        results = results.loc[results['rxn_id']==rxn_id]
    cond = np.array(list(map(lambda x: x.size,list(results['meas_flux'].values))))
    cond_names = results.loc[cond==max(cond),'meas_flux'].iloc[0].columns.values
    heat_mat = pd.DataFrame(columns=cond_names)
    for i in list(results.index.values):
        add = pd.DataFrame(results['lik_cond'][i],columns=results['meas_flux'][i].columns.values,index=[i])
        heat_mat = heat_mat.append(add)
    fig, ax = plt.subplots()
    ax = heatmap(heat_mat,cmap='jet',xticklabels=cond_names)
    ax.set_xlabel('Conditions')
    if rxn_id is not None:
        ax.set_yticklabels(results['regulator'].values[::-1],rotation = 0, ha="right")
        ax.set_title(str('%s: Fit likelihood across conditions' % rxn_id))
        if save:
            fig.savefig(save_dir+rxn_id+'_heat.pdf', bbox_inches='tight')
    else:
        ax.set_title('Fit likelihood across conditions')
        if save:
            fig.savefig(save_dir+'all_heatmap.pdf', bbox_inches='tight')
    plt.show()
    

#%% Plot predicted and measured fluxes
# Plot predicted and measured fluxes across conditions.
# Inputs: index or reaction id, results dataframe, summary dataframe, standard deviation of fluxes.
    
def plot_fit(idx,results,fluxes_sd=None,save=False,save_dir=''):
    if isinstance(idx,int):
        react = results.iloc[[idx]]
        width = 0.4
    elif isinstance(idx,str):
        react = results.loc[results['rxn_id']==idx][::-1]
        width = 0.8/(len(react)+1)
    meas_flux = react['meas_flux']
    pred_flux = react['pred_flux'].values
    sizes = list(map(lambda x:react['meas_flux'].iloc[x].shape[1],list(np.arange(len(react['meas_flux'])))))
    ind = np.arange(max(sizes))
    fig, ax = plt.subplots()
    if fluxes_sd is None:
        plt.bar(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), width, color='r')
    else:
        meas_flux_sd = fluxes_sd.loc[react['rxn_id'].iloc[0],meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns]
        plt.bar(ind, meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].values.reshape(ind.shape), width, color='r', yerr=meas_flux_sd)
    if isinstance(idx,int):
        plt.bar(ind + width, pred_flux[0][0].reshape(ind.shape), width, color='y')
        plt.legend(['Measured', 'Predicted'])
        ax.set_title('%s%s: Flux fit between predicted and measured data' % (results['rxn_id'][idx],results['regulator'][idx]))
    elif isinstance(idx,str):
        colors = cm.summer(np.arange(len(react))/len(react))
        for i in range(len(react)):
            if len(pred_flux[i][0]) < max(sizes):
                formated = np.array([0.0]*max(sizes))
                allcond = list(meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns)
                bools = np.array([False]*max(sizes))
                for j,cond in enumerate(allcond):
                    if any(cond in s for s in list(meas_flux.iloc[i].columns)):
                        bools[j]=True
                np.place(formated,bools,pred_flux[i][0])
                plt.bar(ind + width*(1+i), formated.reshape(ind.shape), width, color = colors[i])
            else:
                plt.bar(ind + width*(1+i), pred_flux[i][0].reshape(ind.shape), width, color = colors[i])
        plt.legend(['Measured']+list(react['regulator'].values))
        ax.set_title('%s: Flux fit between predicted and measured data' % (react['rxn_id'].iloc[0]))
    ax.set_ylabel('Flux (mmol*gCDW-1*h-1)')
    ax.set_xticks(ind + 0.8 / 2)
    ax.set_xticklabels(list(meas_flux.loc[np.array(sizes)==max(sizes)].iloc[0].columns),rotation = 30, ha="right")
    if save:
        fig.savefig(save_dir+'barflux_'+str(idx)+'.pdf', bbox_inches='tight')
    plt.show()
    
#%% Plot improvement of likelihood in best condition
# Plot likelihood improvement 
# Inputs: results dataframe, reaction id.
    
def plot_likelihood(results, cond=None, save=False, save_dir=''):
    noreg = results.loc[results['regulator']==''].reset_index(drop=True)
    if isinstance(cond,str):
        bool_cond = np.array(list(map(lambda x: any(cond in s for s in list(noreg['meas_flux'].iloc[x].columns)),list(np.arange(len(noreg))))))
        noreg = noreg.loc[bool_cond].reset_index(drop=True)
        bottom = np.array(list(map(lambda x: noreg['lik_cond'].iloc[x][0][cond==noreg['meas_flux'].iloc[x].columns][0],list(np.arange(len(noreg))))))
        top = np.array([0.0]*len(bottom))
        for i,rxn in enumerate(list(noreg['rxn_id'].values)):
            rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True)
            bool_rxn = np.array(list(map(lambda x: any(cond in s for s in list(rxn_results['meas_flux'].iloc[x].columns)),list(np.arange(rxn_results.shape[0])))))
            rxn_results = rxn_results[bool_rxn].reset_index(drop=True)
            if (rxn_results.empty==False):
                lik_values = list(map(lambda x: rxn_results['lik_cond'].iloc[x][0][cond==rxn_results['meas_flux'].iloc[x].columns][0],list(np.arange(len(rxn_results)))))
                if bottom[i]<max(lik_values):
                    top[i] = max(lik_values)-bottom[i]
        xlabel = 'Reaction'
        title = str('Likelihood improvement in condition %s' % (cond))
        xticklabels = list(noreg['rxn_id'].values)
        ind = np.arange(len(bottom))
        fig, ax = plt.subplots()
        width = 0.4
        xticks = ind + 0.2
        plt.bar(ind, bottom-min(bottom)+0.2, width, bottom=min(bottom)-0.2, color='r')
        plt.bar(ind, top, width, bottom=bottom, color='y')
    else:
        min_value = np.min(np.concatenate(results['lik_cond'].values,1))
        ncond = np.array(list(map(lambda x: noreg['meas_flux'].iloc[x].shape[1],list(np.arange(len(noreg))))))
        conds = list(noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        ind = np.arange(noreg.shape[0])
        top = pd.DataFrame(index=noreg['rxn_id'],columns=noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        bottom = pd.DataFrame(index=noreg['rxn_id'],columns=noreg['meas_flux'].iloc[ncond==max(ncond)][0].columns)
        fig, ax = plt.subplots()
        title = 'Likelihood improvement'
        width = 0.8/len(conds)
        xticks = ind + 0.8 / 2
        for j,cond in enumerate(conds):
            bool_cond = np.array(list(map(lambda x: any(cond in s for s in list(noreg['meas_flux'].iloc[x].columns)),list(np.arange(len(noreg))))))
            noreg2 = noreg.loc[bool_cond].reset_index(drop=True)
            bottom.loc[bool_cond,cond] = np.array(list(map(lambda x: noreg2['lik_cond'].iloc[x][0][cond==noreg2['meas_flux'].iloc[x].columns][0],list(np.arange(len(noreg2))))))
            for i,rxn in enumerate(list(noreg2['rxn_id'].values)):
                rxn_results = results.loc[(results['rxn_id']==rxn)&(results['regulator']!='')].reset_index(drop=True)
                bool_rxn = np.array(list(map(lambda x: any(cond in s for s in list(rxn_results['meas_flux'].iloc[x].columns)),list(np.arange(rxn_results.shape[0])))))
                rxn_results = rxn_results[bool_rxn].reset_index(drop=True)
                if (rxn_results.empty==False):
                    lik_values = list(map(lambda x: rxn_results['lik_cond'].iloc[x][0][cond==rxn_results['meas_flux'].iloc[x].columns][0],list(np.arange(len(rxn_results)))))
                    if bottom.loc[rxn,cond]<max(lik_values):
                        top.loc[rxn,cond] = max(lik_values)-bottom.loc[rxn,cond]
            plt.bar(ind+width*(j), bottom[cond].values-min_value+0.2, width, bottom=min_value-0.2, color='r')
            plt.bar(ind+width*(j), top[cond].values, width, bottom=bottom[cond].values, color='y')
        
    xlabel = 'Reaction'
    xticklabels = list(noreg['rxn_id'].values)
    ax.set_ylabel('Likelihood')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,rotation = 45, ha="right")
    plt.legend(['General M-M','1 Regulator'],loc='upper left')
    if save:
        fig.savefig(save_dir+'improvement_'+str(cond)+'.pdf', bbox_inches='tight')
    plt.show()
    