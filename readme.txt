RbRFit Modeling
Check availability of data: Check_availability.ipynb
Import data as pandas dataframe
Define reactions: 
summary = arf.define_reactions(rxn_id,model,fluxes,proteins,metabolites)
Define candidates:
reg_coli = pd.read_csv(data_dir+"SMRN.csv",index_col="rxn_id")
candidates = arf.define_candidates(rxn_id,reg_coli,metabolites)
Modeling:
markov_par = {'freq':20,'nrecord':200,'burn_in':0}
results = arf.fit_reactions(summary,model,markov_par,candidates)
Validation:
val_bycond, val_results = arf.validate_bycond(results,summary,candidates=candidates)
Multi-layer prioritization score:
shortlist = arf.rankresults(val_results,model)

RbRFit Modeling in server:
python 1_load_data.py
python 2_fit_reactions.py model_path
sh run_fit_reaction.txt # if too many commands, sparate file in several batches
python 3_stack_outputs.py number_of_reactions results_all.pickle
python 4_validation.py

Troubleshooting:
for i in `grep -m 1 -l Exited lsf.o*` ; do ((count = 151 - `grep Running $i | wc -l`)); grep -h -o "python fit_reaction.py [0-9]* [0-9]* [./A-Z0-9a-z_]*;" $i | tail -$count; done > failed.txt
python troubleshoot.py failed.txt