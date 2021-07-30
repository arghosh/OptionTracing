import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys

create_dirs = ['logs/', 'slurm/', 'configs/']
for c in create_dirs:
    if not os.path.exists(c):
        try:
            os.makedirs(c)
        except FileExistsError:
            pass

def get_memory(combo):
    if combo['setup'] =='cf':
        memory_map = {'coda': 60000, 'ednet': 75000, 'eedi': 20000}
    else:
        memory_map = {'coda': 45000, 'ednet': 68000, 'eedi': 20000}
    return memory_map[combo['dataset']]

def get_cpu(combo):
    return 4

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def get_run_id():
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts)
    return run_id

#CF 
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
#KT
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

def get_gpu(combo):
    if combo['model']=='bi-gcn':
        return "2080ti"
    if combo['model']in {'bi-lstm', 'dkvmn', 'attn-akt'}:
        return "1080ti"
    else:
        return "titanx"

    
def is_valid(combo):
    return True
    



other_dependencies = {'gpu': get_gpu, 'memory': get_memory, 'n_cpu':get_cpu, 'valid':is_valid}

run_id = int(get_run_id())

key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
gpu_counts =collections.defaultdict(int)

for combo in combinations:
    # Write the scheduler scripts
    with open("template.sh", 'r') as f:
        schedule_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
         combo[k] = v(combo)
    if not combo['valid']:
        #print(combo)
        continue
    combo['run_id'] = run_id
    gpu_counts[combo['gpu']] +=1

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))
    


    schedule_script += "\n"

    # Write schedule script
    script_name = 'configs/cv_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name +", Time Now= "+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" 
    with open("logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

print(gpu_counts)
# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    #print(command)
    print(subprocess.check_output(command, shell=True))
