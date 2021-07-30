# Option Tracing: Beyond Correctness Analysis in Knowledge Tracing

This is an official repository of the paper [Option Tracing: Beyond Correctness Analysis in Knowledge Tracing](https://arxiv.org/abs/2104.09043), presented at AIED 2021. 


## Environment Setup
This repository uses the following packages in Python3.
```
torch==1.7.1
```

## Training
I added sample datasets in the `data/` folder. Download the full preprocessed Ednet and Eedi (renamed as Coda) dataset from [Google Drive](). 

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