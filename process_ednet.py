import numpy as np
import json
import time
from utils import dump_json, open_json
import pandas as pd
from multiprocessing import Pool
import math
import argparse
import os
from utils import format_data_as_kt
RAW_DIR = '../data/KT1/'
abcd_map = {'a':1, 'b':2, 'c':3, 'd':4}
q_map = {}
convert_mapper = {}
def f(name):
    global q_map, abcd_map
    path = RAW_DIR+name
    user_id = name.split('.')[0][1:]
    user_df = pd.read_csv(path).sort_values('timestamp')
    q_ids, correct_ans, ans, labels, times, quiz_ids = [], [], [], [], [], []
    subject_ids = []
    last_timestamp = None
    for _, row in user_df.iterrows():
        try:
            ans.append(abcd_map[row['user_answer']])
            
        except:
            print(user_id)
            print(row)
            continue
        q_ids.append(int(row['question_id'][1:]))
        correct_ans.append(
            q_map[int(row['question_id'][1:])]['correct_ans'])
        
        
        
        if correct_ans[-1]==ans[-1]:
            labels.append(1)
        else:
            labels.append(0)
        
        if len(times) > 0:
            times.append(   (int(row['timestamp']) - last_timestamp)/86400000.) 
        else:
            times.append(0.)
        last_timestamp = row['timestamp']
    subject_ids = [q_map[d]['tags'] for d in q_ids]
    quiz_ids = [q_map[d]['bundle_id'] for d in q_ids]

    out = {'user_id': int(user_id), 'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans,
           'ans': ans, 'labels': labels, 'times': times, 'quiz_ids': quiz_ids}
    return out
def f2(d):
    schema = ['subject_ids', 'q_ids', 'correct_ans', 'ans', 'labels']
    global convert_mapper
    temp_subjects =  []
    for ds in d['subject_ids']:
        temp = [convert_mapper['tag_id'][dss] for dss in ds]
        temp_subjects.append(temp)
    d['subject_ids'] = temp_subjects
    d['q_ids'] = [convert_mapper['q_id'][dss] for dss in d['q_ids']]
    temp_d = {}
    for field in schema:
        temp_d[field] = d[field]
    d = temp_d
    return d

def main():
    global q_map
    question_df =pd.read_csv('../data/contents/questions.csv')
    for _, row in question_df.iterrows():
        q_id = int(row['question_id'][1:])
        correct_answer = abcd_map[row['correct_answer']]
        tags = row['tags'].split(';')
        tags = [int(d) for d in tags]
        bundle = int(row['bundle_id'][1:])
        q_map[q_id] = {'correct_ans': correct_answer, 'tags':tags, 'bundle_id':bundle}
    #res = f('u560.csv')
    file_names = os.listdir(RAW_DIR)
    with Pool(30) as p:
        results = p.map(f, file_names)
    dump_json('data/ednet.json', results)
    dump_json('data/ednet_sample.json', results[:100])
    dump_json('data/q_map.json', q_map)
    
def main2():
    global q_map, convert_mapper
    q_map = open_json('data/q_map.json')
    convert_mapper = {'q_id':{}, 'tag_id':{}}
    for q, v in q_map.items():
        convert_mapper['q_id'][int(q)] =  len(convert_mapper['q_id'])
        v['new_q_id'] = convert_mapper['q_id'][int(q)]
        v['new_tag_id'] = []
        for t in v['tags']:
            if t not in convert_mapper['tag_id']:
                convert_mapper['tag_id'][t] = len(convert_mapper['tag_id'])
            v['new_tag_id'].append(convert_mapper['tag_id'][t])
    dump_json('data/convert_mapper.json', convert_mapper)
    data = open_json('data/ednet.json')
    with Pool(30) as p:
        results = p.map(f2, data)
    dump_json('data/ednet_converted.json', results)
    dump_json('data/ednet_converted_sample.json', results[:100])

def f3(d):
    return format_data_as_kt(d, 200)

def main3():
    data = open_json('data/ednet_converted.json')
    with Pool(30) as p:
        results = p.map(f3, data)
        results = [ds for d in results for ds in d]
    dump_json('data/kt_ednet.json', results)

    data = open_json('data/coda.json')
    with Pool(30) as p:
        results = p.map(f3, data)
        results = [ds for d in results for ds in d]
    dump_json('data/kt_coda.json', results)

def f4(d):
    output = format_data_as_kt(d, 1000)
    for out in output:
        out['test_mask'] = [1]*len(out['q_ids'])
    return output

def main4():
    data = open_json('data/ednet_converted.json')
    with Pool(30) as p:
        results = p.map(f4, data)
    all_results = []
    for idx, ds in enumerate(results):
        for d in ds:
            d["user_id"] =  idx
            all_results.append(d)
    print(len(all_results))
    dump_json('data/cf_ednet.json', all_results)


def f5(ds):
    N = len(ds['q_ids'])
    ds['dash_features'] = []
    for idx in range(N):
        wrong_before = math.log(
            1+len([d for d in range(idx) if  ds['labels'][d] == 0]))
        correct_before = math.log(
            1+len([d for d in range(idx) if  ds['labels'][d] == 1]))
        wrong_same_before = math.log(
            1+len([d for d in range(idx) if  ds['labels'][d] == 0 and len(
                set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
        correct_same_before = math.log(
            1+len([d for d in range(idx) if  ds['labels'][d] == 1 and len(
                set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))

        wrong_after = math.log(
            1+len([d for d in range(idx+1, N) if  ds['labels'][d] == 0]))
        correct_after = math.log(
            1+len([d for d in range(idx+1, N) if  ds['labels'][d] == 1]))
        wrong_same_after = math.log(
            1+len([d for d in range(idx+1, N) if  ds['labels'][d] == 0 and len(
                set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
        correct_same_after = math.log(
            1+len([d for d in range(idx+1, N) if ds['labels'][d] == 1 and len(
                set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
        ds['dash_features'].append(
            [wrong_before, correct_before, wrong_same_before, correct_same_before, wrong_after, correct_after, wrong_same_after, correct_same_after])

    return ds

def main5():
    data = open_json('data/cf_ednet.json')
    with Pool(30) as p:
        results = p.map(f5, data)
    dump_json('data/cf_ednetdash.json', results)
    

def convert_eedi():
    data = open_json('data/eedi.json')
    with Pool(3) as p:
        results = p.map(f3, data)
        results = [ds for d in results for ds in d]
    dump_json('data/kt_eedi.json', results)

def create_eedi_map():
    data = open_json('data/question_metadata_task_3_4.json')
    output = {'qs':{}, 'sub':{}}
    for q,v in data.items():
        question = int(q)
        subjects = v['child_map']
        output['qs'][question] = subjects
        for d in subjects:
            if d not in output['sub']:
                output['sub'][d] = []
            output['sub'][d].append(question)
    max_len =0
    for k,v in output['qs'].items():
        max_len = max(max_len, len(v))
    print(max_len, len(output['qs']))
    #dump_json('data/eedi_subject_to_question_map.json', output)
    




if __name__ == "__main__":
    data = open_json('data/eedi.json')
    print(len(data))
    max_id = 0
    for d in data:
        max_id = max(max_id, d['user_id'])
    print(max_id)
    #convert_eedi()
    #create_eedi_map()
    # remove_unknowns = False
    # convert_kt = False
    # convert_cf = False
    # convert_dash = True
    # if remove_unknowns:
    #     main2()
    # elif convert_kt:
    #     main3()
    # elif convert_cf:
    #     main4()
    # elif convert_dash:
    #     main5()
    # else:
    #     main()
    
