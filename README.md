# Option Tracing: Beyond Correctness Analysis in Knowledge Tracing

This is an official repository of the paper [Option Tracing: Beyond Correctness Analysis in Knowledge Tracing](https://arxiv.org/abs/2104.09043), presented at AIED 2021. 


## Environment Setup
This repository uses the following packages in Python3.
```
torch==1.7.1
```

## Training
I added sample datasets in the `data/` folder. Download the full preprocessed Ednet and Eedi (renamed as Coda) dataset from [Google Drive](https://drive.google.com/file/d/1sPBGd6atvseSvr7rqzuexWitQk_TrVM1/view?usp=sharing). 

```(bash)
python {train_file}\
    --lr {lr}\
    --setup {setup}\
    --task {task}\
    --fold {fold}\
    --batch_size {batch_size}\
    --dataset {dataset}\
    --model {model}\
    --hidden_dim {hidden_dim}\
    --question_dim {question_dim}\
    --neptune --cuda
```
For CF setup use the following parameters to run,
```(bash)
hyperparameters = [
    [('setup',), ['cf']],
    [('lr',), [1e-4]],
    [('train_file',), ['cf_trainer.py']],
    [('task',), [2]],
    [('fold',), [1,2,3,4,5]],
    [('batch_size',), [64]],
    [('question_dim',), [32,64]],
    [('hidden_dim',), [256]],
    [('dataset',), ['coda', 'ednet']],
    [('model',), ["bi-gcn", "bi-lstm", 'ncf', 'lstm']],
]
```
For KT setup use the following parameters to run,
```(bash)
hyperparameters = [
    [('setup',), ['kt']],
    [('lr',), [1e-4]],
    [('train_file',), ['kt_trainer.py']],
    [('task',), [2]],
    [('fold',), [1,2,3,4,5]],
    [('batch_size',), [64]],
    [('question_dim',), [64,128]],
    [('hidden_dim',), [512]],
    [('dataset',), ['coda', 'ednet']],
    [('model',), ["attn-akt", "attn", 'dkvmn', 'lstm']],
]
```

## Data Format
There are four partitions for each student's sequential response. This is a rather unfortunate name: `test_mask` for a single time step is 0 if the answer is unobserved (happens in some dataset) or that time step is a padding time step (to make all student's length same); otherwise, `test_mask` is 1. `valid_mask` for a single time step is 0 if that timestep is part of validation set. `local_test_mask` for a single time step is 0 if that timestep is part of the Test set. Thus, training is done over `test_mask*valid_mask*local_test_mask` that denotes not padding timepoint, not validation timepoint, not testing timepoint.

```(bash)
data = open_json(data/cf_coda.json)`: List(Dict), a list of student response data.

data[idx]: Dict, a single student's  sequential response data. Example:
{
`user_id`: 1,# Id of the user
`subject_ids`: [[5], [2,3], [4], [2], [0]],# List(List(subject ids))
`q_ids`:[1,2,3,4,0], #List(int), question indices
`correct_ans`: [0,1,2,3,0], #List(int), correct options
`ans`:[0,1,3,0,0],#List(int), #List(int), student answeres
`labels`:[1,1,0,0,0], #List(int), binary correctness
`test_mask`: [1,1,1,1,0] #List(int), 0 if unknown (due to padding or unobserved)
# These two are generated in the script.
'valid_mask`:[0,1,1,1,1], #List(int), 0 if it is part of validation set
'local_test_mask`: [1,0,1,1,1] #List(int), 0 if it is part of test set
}

```


## Citation
If you find this code useful in your research then please cite  
```(bash)
@inproceedings{ghosh2021option,
  title={Option Tracing: Beyond Correctness Analysis in Knowledge Tracing},
  author={Ghosh, Aritra and Raspat, Jay and Lan, Andrew},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={137--149},
  year={2021},
  organization={Springer}
}
``` 

Contact:  Aritra Ghosh (aritraghosh.iem@gmail.com).
