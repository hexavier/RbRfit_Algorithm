# -*- coding: utf-8 -*-

#%% Import modules
import pandas as pd
import sympy as sym

#%% Define reactions
# For each of the reactions, the function creates a data frame where every row constitutes a reaction.
# Inputs: list of reaction ids that will be analyzed, stoichiometric model, DataFrame containing fluxes x conditions,
# DataFrame containing prot x cond, and DataFrame with metabolites x cond.

def define_reactions(rxn_id, model, fluxes, prot, metab):
    from numpy import isnan
    mapping = pd.read_table("ECOLI_83333_idmapping.dat",header=None)
    reaction, reactant, product, enzyme, flux = ([] for l in range(5))
    for i in range(len(rxn_id)):
        # Reaction value
        reaction.append(model.reactions.get_by_id(rxn_id[i]).reaction)
        # Reactant values
        react = model.reactions.get_by_id(rxn_id[i]).reactants
        react_df = []
        name_r = []
        ncond = fluxes.shape[1]
        react_bool = [True]*ncond
        for j in range(len(react)):
            met_r = react[j].id[:-2] #strip compartment letter from id
            if (any(met_r in s for s in ['h','h2o'])==0):
                react_df.append(metab.loc[met_r].values)
                name_r.append(react[j].id)
                react_bool = (react_bool & isnan(metab.loc[met_r].values)==0)
        react_df = pd.DataFrame(react_df,columns = metab.columns, index = name_r)
        # Product values
        prod = model.reactions.get_by_id(rxn_id[i]).products
        prod_df = []
        name_p = []
        prod_bool = [True]*ncond
        for j in range(len(prod)):
            met_p = prod[j].id[:-2] #strip compartment letter from id
            if (any(met_p in s for s in [metab.index.values]) and (any(met_p in s for s in ['h','h2o'])==0)):
                prod_df.append(metab.loc[met_p].values)
                name_p.append(prod[j].id)
                prod_bool = (prod_bool & isnan(metab.loc[met_p].values)==0)
        prod_df = pd.DataFrame(prod_df,columns = metab.columns, index = name_p)
        # Enzyme values
        enz = list(model.reactions.get_by_id(rxn_id[i]).genes)
        enz_df = []
        name_e = []
        enz_bool = [True]*ncond
        for j in range(len(enz)):
            gene = mapping[mapping[2]==enz[j].id][0].reset_index()
            gene = list(mapping[((mapping[0]==gene[0][0]) & (mapping[1]=='Gene_Name'))][2])
            if any(gene[0] in s for s in [prot.index.values]):
                enz_df.append(prot.loc[gene[0]].values)
                name_e.append(gene)
                enz_bool = (enz_bool & isnan(prot.loc[gene[0]].values)==0)
        enz_df = pd.DataFrame(enz_df, columns = prot.columns, index = name_e)
        # Append all data
        flux_bool = isnan(fluxes.loc[rxn_id[i]].values)==0
        bool_all = (react_bool & prod_bool & enz_bool & flux_bool)
        reactant.append(react_df.loc[:,bool_all])
        product.append(prod_df.loc[:,bool_all])
        enzyme.append(enz_df.loc[:,bool_all])
        flux.append(pd.DataFrame([fluxes.loc[rxn_id[i]].values],columns = fluxes.columns, index = [rxn_id[i]]).loc[:,bool_all])
        
        
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
            # Reactant values
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
            act_coli.append(act_coli_df)
            inh_coli.append(inh_coli_df)
        else:
            act_coli.append([])
            inh_coli.append([])
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
            # parameters.append(K), species.append(enz), speciestype.append('enz')
            
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
        expr = {'full':vmax*(num/den),'occu':(num/den)}
        
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
    from numpy import log2,nanmedian#,nan
    reaction = model.reactions.get_by_id(summary['rxn_id'][idx])
    distribution, par1, par2 = ([] for i in range(3))
    for i,par in enumerate(param['parameters']):
        if param['speciestype'][i] == 'met':
            distribution.append('unif')
            if any(param['species'][i] in s for s in [summary['reactant'][idx].index.values]):
                par1.append(-15.0+log2(nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+log2(nanmedian(summary['reactant'][idx].loc[param['species'][i]].values)))
            elif any(param['species'][i] in s for s in [summary['product'][idx].index.values]):
                par1.append(-15.0+log2(nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
                par2.append(15.0+log2(nanmedian(summary['product'][idx].loc[param['species'][i]].values)))
#         elif param['speciestype'][i] == 'enz':
#             distribution.append(nan)
#             par1.append(nan)
#             par2.append(nan)
        elif param['speciestype'][i] == 'K_eq':
            distribution.append('unif')
            Q_r = 1
            for subs in list(summary['reactant'][idx].index):
                Q_r /= (summary['reactant'][idx].loc[subs].values)**abs(reaction.get_coefficient(subs))
            products = list(summary['product'][idx].index.values)
            if products:
                for prod in products:
                    Q_r *= (summary['product'][idx].loc[prod].values)**abs(reaction.get_coefficient(prod))
                par1.append(-20.0+log2(nanmedian(Q_r)))
                par2.append(20.0+log2(nanmedian(Q_r)))
    param['distribution'] = pd.Series(distribution, index=param.index)
    param['par1'] = pd.Series(par1, index=param.index)
    param['par2'] = pd.Series(par2, index=param.index)
    return param

#%% Draw parameters
# From the prior distribution, update those parameters that are present in ‘updates’.
# Inputs: parameter indeces to update within a list, parameter dataframe with priors, current values.
def draw_par(update, parameters, current):
    from numpy.random import uniform
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
    import numpy as np
    from scipy.stats import norm
    from scipy.optimize import nnls
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
        kcat, residual = nnls((pred_occu*enz).reshape((ncond,1)), flux.reshape(ncond))
        pred_flux = kcat*pred_occu*enz
        npars = len(current)+len(kcat)
        var = np.sum((flux-pred_flux)**2)/(ncond-npars)
        likelihood = norm.pdf(flux, pred_flux, np.sqrt(var))
        return np.sum(np.log(likelihood))

#%% Fit reaction equation using MCMC-NNLS
# Sample posterior distribution Pr(Ω|M,E,jF) using MCMC-NNLS.
# Inputs: markov parameters (fraction of samples that are reported, how many samples are 
# desired, how many initial samples are skipped), parameters table with priors, 
# summary as generated in define_reactions, equations.
def fit_reaction_MCMC(idx, markov_par, parameters, summary, equations, candidates=None):
    from numpy.random import uniform
    from numpy import inf,exp
    print('Running MCMC-NNLS for reaction %d.' % idx)
    colnames = list(parameters['parameters'].values)
    colnames.append('likelihood')
    track = pd.DataFrame(columns=colnames)
    current_pars = [None] * len(parameters)
    current_pars = draw_par([p for p in range(len(parameters))], parameters, current_pars)
    current_lik = calculate_lik(idx, parameters, current_pars, summary, equations)
    for i in range(markov_par['burn_in']+markov_par['nrecord']*markov_par['freq']):
        for p in range(len(parameters)):
            proposed_pars = draw_par([p], parameters, current_pars)
            proposed_lik = calculate_lik(idx, parameters, proposed_pars, summary, equations)
            if ((uniform(0,1) < exp(proposed_lik-current_lik)) or \
                (proposed_lik==current_lik and proposed_lik==-inf)):
                current_pars = proposed_pars
                current_lik = proposed_lik
        if (i > markov_par['burn_in']):
            if ((i-markov_par['burn_in'])%markov_par['freq'])==0:
                add_pars = list(current_pars)
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