# -*- coding: utf-8 -*-

#%% Import modules
import pandas as pd
import sympy as sym
import numpy as np
from numpy.random import uniform
from scipy.stats import norm
from scipy.optimize import nnls

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
                names.append(gene)
                mol_bool = (mol_bool & np.isnan(dataset.loc[gene[0]].values)==0)
    else:
        for j in range(len(molecules)):
            met = molecules[j].id[:-2] #strip compartment letter from id
            if ((mol_type=='reactant')and(any(met in s for s in ['h','h2o'])==0)) or \
            ((mol_type=='product')and(any(met in s for s in [dataset.index.values]) and \
              (any(met in s for s in ['h','h2o'])==0))):
                    mol_df.append(dataset.loc[met].values)
                    names.append(molecules[j].id)
                    mol_bool = (mol_bool & np.isnan(dataset.loc[met].values)==0)
    mol_df = pd.DataFrame(mol_df,columns = dataset.columns, index = names)
    return mol_df, mol_bool
    
#%% Define reactions
# For each of the reactions, the function creates a data frame where every row constitutes a reaction.
# Inputs: list of reaction ids that will be analyzed, stoichiometric model, DataFrame containing fluxes x conditions,
# DataFrame containing prot x cond, and DataFrame with metabolites x cond.

def define_reactions(rxn_id, model, fluxes, prot, metab):
    reaction, reactant, product, enzyme, flux = ([] for l in range(5))
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
        
    # Provisional binding site before we can automatize it:
    binding_site = [[['fum_c','mal_L_c']],[['6pgc_c'],['nadp_c','nadph_c']]\
                    ,[['mal_L_c'],['nad_c','nadh_c']],[['atp_c','adp_c'],['f6p_c','fdp_c']]\
                    ,[['g6p_c','f6p_c']],[['atp_c','amp_c'],['pyr_c','pep_c']]\
                    ,[['adp_c','atp_c'],['pep_c','pyr_c']],[['r5p_c']]]
    
    summary = pd.DataFrame({'idx':range(len(rxn_id)),'reaction':reaction,'rxn_id':rxn_id,\
                            'reactant':reactant,'product':product,\
                            'enzyme':enzyme,'flux':flux,'binding_site':binding_site})
    summary = summary.set_index('idx')
    return summary

#%% Define candidates
# For each reaction, a table with the regulators is created. 
# Inputs: list of reaction ids that will be analyzed, DataFrame with regulators
# for all rxn_id in E. coli, DataFrame with metabolites x cond, and DataFrame with
# regulators for all rxn_id in other organisms (optional).
    
def define_candidates(rxn_id,reg_coli,metab,reg_other=None):
    act_coli, inh_coli, act_other, inh_other = ([] for l in range(4))
    for i in range(len(rxn_id)):
        if (any(rxn_id[i].lower() in s for s in [reg_coli.index.values])):
            cand_coli = reg_coli.loc[rxn_id[i].lower()].reset_index()
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
                act_coli_df.drop_duplicates(inplace=True)
                act_coli.append(act_coli_df)
            if inh_coli_df.empty:
                inh_coli.append('No data available for the candidate activators.')
            else:
                inh_coli_df.drop_duplicates(inplace=True)
                inh_coli.append(inh_coli_df)
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

#%% Write Rate Equations
# For each of the models, write one rate equation expression. If products are available, include them.
# Inputs: summary generated by define_reactions, idx defining the reaction that is analyzed,
# stoichiometric model and candidates dataframe (optional).

def write_rate_equations(idx,summary, model, candidates=None):
    if candidates is None:
        parameters, species, speciestype, vbles = ([] for i in range(4))
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
            parameters.append(K), species.append(sub), speciestype.append('met'), vbles.extend([K,S])
        num1 = 1/num1            
        
        product = list(summary['product'][idx].index)
        if product:
            num3 = sym.sympify('1')
            for prod in product:
                P = str('c_%s' % prod)
                exp = abs(reaction.get_coefficient(prod))
                num3 *= sym.sympify(P+'**'+str(exp))
            K_eq = sym.symbols('K_eq')
            parameters.append('K_eq'), species.append('K_eq'), speciestype.append('K_eq'), vbles.extend([K_eq,P])
            num3 = (1/K_eq)*num3
            num = num1*(num2-num3)
        else:
            num = num1*num2
        
        # Define the denominator:
        den = sym.sympify('1')
        for i,site in enumerate(summary['binding_site'][idx]):
            den_site = sym.sympify('1')
            for met in summary['binding_site'][idx][i]:
                exp = int(abs(reaction.get_coefficient(met)))
                for j in range(1, (exp+1)):
                    R = str('c_%s' % met)
                    K = str('K_%s' % met)
                    den_site += sym.sympify('('+R+'/'+K+')**'+str(j))
                    parameters.append(K), species.append(met), speciestype.append('met'), vbles.extend([R,K])
            den *= den_site
        
        # Paste all the parts together:
        expr = {'vmax':vmax,'occu':(num/den)}
        
        # Generate list of parameters:
        parframe = pd.DataFrame({'parameters':parameters,'species':species,'speciestype':speciestype})
        parframe.drop_duplicates('parameters',inplace=True)
        parframe.reset_index(drop=True,inplace=True)
        
        return expr,parframe, vbles

#%% Build parameter priors
# For each of the parameters, define the prior/proposal distribution needed for MCMC.
# Inputs: dataframe with parameters, summary generated in define_reactions, the 
# stoichiometric model, and candidate dataframe (optional).            
def build_priors(param, idx, summary, model, candidates=None):
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
        elif param['speciestype'][i] == 'K_eq':
            distribution.append('unif')
            Q_r = 1
            for subs in list(summary['reactant'][idx].index):
                Q_r /= (summary['reactant'][idx].loc[subs].values)**abs(reaction.get_coefficient(subs))
            products = list(summary['product'][idx].index.values)
            if products:
                for prod in products:
                    Q_r *= (summary['product'][idx].loc[prod].values)**abs(reaction.get_coefficient(prod))
                par1.append(-20.0+np.log2(np.nanmedian(Q_r)))
                par2.append(20.0+np.log2(np.nanmedian(Q_r)))
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
        else:
            print('Invalid distribution')
    return draw

#%% Calculate likelihood
# Calculate the likelihood given the flux point estimate or the lower and upper bounds of flux variability analysis.
# Inputs: parameter dataframe, current values, summary as generated in define_reactions, 
# equations.
def calculate_lik(idx,parameters, current, summary, equations):
    if len(summary['flux'][idx]) == 1: # fluxes as point estimates
        flux = summary['flux'][idx].values
        occu = equations['occu']
        ncond = flux.shape[1]
        current = np.array(current)
        vbles = []
        vbles_vals = []
        for par in list(parameters['parameters'].values):
            vbles.append(par)
            rep_par = np.repeat(current[parameters['parameters'].values==par],ncond)
            vbles_vals.append(rep_par)
        for sub in list(summary['reactant'][idx].index):
            vbles.append('c_'+sub)
            vbles_vals.append(summary['reactant'][idx].loc[sub].values)
        for prod in list(summary['product'][idx].index):
            vbles.append('c_'+prod)
            vbles_vals.append(summary['product'][idx].loc[prod].values)
            
        f = sym.lambdify(vbles, occu)
        pred_occu = f(*vbles_vals)
        enz = summary['enzyme'][idx].values
        kcat, residual = nnls((pred_occu*enz).reshape((ncond,1)), flux.reshape(ncond)) ##Bug
        pred_flux = kcat*pred_occu*enz
        npars = len(current)+len(kcat)
        var = np.sum((flux-pred_flux)**2)/(ncond-npars)
        likelihood = norm.pdf(flux, pred_flux, np.sqrt(var))
        return np.sum(np.log(likelihood)),kcat

#%% Fit reaction equation using MCMC-NNLS
# Sample posterior distribution Pr(Ω|M,E,jF) using MCMC-NNLS.
# Inputs: markov parameters (fraction of samples that are reported, how many samples are 
# desired, how many initial samples are skipped), parameters table with priors, 
# summary as generated in define_reactions, equations.
def fit_reaction_MCMC(idx, markov_par, parameters, summary, equations, candidates=None):
    print('Running MCMC-NNLS for reaction %d...' % idx)
    colnames = list(parameters['parameters'].values)
    colnames.append('kcat')
    colnames.append('likelihood')
    track = pd.DataFrame(columns=colnames)
    current_pars = [None] * len(parameters)
    current_pars = draw_par([p for p in range(len(parameters))], parameters, current_pars)
    current_lik,cur_kcat = calculate_lik(idx, parameters, current_pars, summary, equations)
    for i in range(markov_par['burn_in']+markov_par['nrecord']*markov_par['freq']):
        for p in range(len(parameters)):
            proposed_pars = draw_par([p], parameters, current_pars)
            proposed_lik,pro_kcat = calculate_lik(idx, parameters, proposed_pars, summary, equations)
            if ((uniform(0,1) < np.exp(proposed_lik)/(np.exp(proposed_lik)+np.exp(current_lik))) or \
                (proposed_lik > current_lik)):
                current_pars = proposed_pars
                cur_kcat = pro_kcat
                current_lik = proposed_lik
        if (i > markov_par['burn_in']):
            if ((i-markov_par['burn_in'])%markov_par['freq'])==0:
                add_pars = list(current_pars)
                add_pars.append(float(cur_kcat))
                add_pars.append(current_lik)
                track = track.append(pd.DataFrame([add_pars],columns=colnames))
    track.reset_index(drop=True,inplace=True)
    return track

#%% Fit reaction equations
# Run all required functions as a block to fit predicted to measured flux.
# Inputs: summary dataframe, stoichiometric model, markov parameters, and candidates dataframe (optional).

def fit_reactions(summary,model,markov_par,candidates=None):
    results = pd.DataFrame(columns=['idx','reaction','rxn_id','reg_type','regulator','best_fit','best_lik'])
    for idx in list(summary.index):
        expr,parameters,vbles = write_rate_equations(idx,summary,model)
        parameters = build_priors(parameters,idx,summary,model)
        track = fit_reaction_MCMC(idx,markov_par,parameters,summary,expr)
        max_lik = max(track['likelihood'].values)
        max_par = track[track['likelihood'].values==max_lik]
        add = {'idx':idx,'reaction':summary['reaction'][idx],'rxn_id':summary['rxn_id'][idx],\
               'reg_type':[''],'regulator':[''],'best_fit':max_par,'best_lik':max_lik}
        results = results.append([add])
    results = results.set_index('idx')
    results = results.sort_values(by='best_lik')
    return results