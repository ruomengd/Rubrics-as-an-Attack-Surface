# Rubrics-as-an-Attack-Surface


## data 


## biased rubric search


## downstream evaluation

### dpo training
run `train_dpo.sh` to train your policy with training data as well as their labels by the given rubrics.

### policy evaluation
run 5 steps at your choice in `eval.sh`, to:
 - generate responses, 
 - score them, 
 - analyze win-rate, 
 - select bon response, 
 - eval by 3rd party judge.
