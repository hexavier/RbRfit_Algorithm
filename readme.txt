RbRFit Modeling

python 1_load_data.py
python 2_fit_reactions.py model_path
sh run_fit_reaction.txt # if too many commands, sparate file in several batches
python 3_stack_outputs.py number_of_reactions results_all.pickle
python 4_validation.py

Troubleshooting
for i in `grep -m 1 -l Exited lsf.o*` ; do ((count = 151 - `grep Running $i | wc -l`)); grep -h -o "python fit_reaction.py [0-9]* [0-9]* [./A-Z0-9a-z_]*;" $i | tail -$count; done > failed.txt
python troubleshoot.py failed.txt