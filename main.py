import os
import json
import zipfile
from pprint import pprint
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--code_path', '-c', default="./1112_H210301-PE5-20230426", type=str)
parser.add_argument('--chatgpt_path', '-g', default=['chatgpt1.py', 'chatgpt2.py', 'chatgpt3.py'], action='append')
parser.add_argument('--model', '-m', default='codebert', type=str)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_OF_CODE = args.code_path
PATH_OF_GPT = [f"{PATH_OF_CODE}/{i}" for i in args.chatgpt_path]
MODEL = args.model

def py_text_preprocessing(py_path):
    py_text= ""
    with open(py_path) as f:
        ls_line = f.readlines()
        for i, line in enumerate(ls_line):
            # line = line.strip() # remove \t, space
            if '#' in line:
                # remove comment
                line = line.split('#')[0]
            if len(line) != 0:
                py_text = py_text + line + '\n'
    return py_text

def preprocessing_student():
    ls_student = []
    for item in tqdm(os.listdir(PATH_OF_CODE), desc='[Prepprocessing Student Data]'):
        if os.path.isdir(os.path.join(PATH_OF_CODE, item)):
            s_id, s_name, _ = item.split('_')
            s_id, s_name = s_id.strip(), s_name.strip().replace('　', '').replace(' ', '')

            ls_zip = glob(os.path.join(PATH_OF_CODE, item) + '/*.zip')
            
            if len(ls_zip) != 0:
                # 有 .zip 檔案
                with zipfile.ZipFile(ls_zip[0]) as zip:
                    zip.extractall(os.path.dirname(ls_zip[0]))

            ls_py = glob(os.path.join(PATH_OF_CODE, item) + '/*.py') + glob(os.path.join(PATH_OF_CODE, item) + '/**/*.py')
            if len(ls_py) == 0:
                print(f"{s_name} 沒有 .py 檔案")
            elif len(ls_py) > 1:
                print(f"{s_name} 有多個 .py 檔案")

            if len(ls_py) >= 1:                
                s_py_path = ls_py[0]
            s_py_txt = py_text_preprocessing(s_py_path)
            ls_student.append(dict(id = s_id,
                                   name = s_name,
                                   py_txt = s_py_txt))

    with open(f'{PATH_OF_CODE}/student_codes.json', 'w') as f:
        json.dump(ls_student, f)

    return ls_student

def preprocessing_gpt():
    ls_gpt = []
    for i, gpt_path in tqdm(enumerate(PATH_OF_GPT), desc='[Preprocessing ChatGPT Data]'):
        gpt_py_txt = py_text_preprocessing(gpt_path)
        ls_gpt.append(dict(name=f'chatgpt',
                           id=str(i+1),
                           py_txt=gpt_py_txt))
    return ls_gpt

def get_code_embedding(ls, model, tokenizer):
    ls_py_txt = []
    for student in ls:
        ls_py_txt.append(student['py_txt'])

    encoder_input = tokenizer(ls_py_txt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    embeddings = model(encoder_input['input_ids'].to(DEVICE))[0][:, :, :]
    print('embedding shape:' ,embeddings.shape)
    embeddings = embeddings.detach().cpu()
    return embeddings

def detect_sim_between_students(ls_student, ls_s_embedding):
    ls_sim_pair = []
    for i in tqdm(range(len(ls_student) - 1), desc='[Calculate Students Similarity]'):
        for j in range(i + 1, len(ls_student)):
            ls_sim_pair.append([
                f"{ls_student[i]['id']}_{ls_student[i]['name']}",
                f"{ls_student[j]['id']}_{ls_student[j]['name']}",
                F.cosine_similarity(ls_s_embedding[i].flatten(), ls_s_embedding[j].flatten(), dim=-1).numpy()])
    pd.DataFrame(ls_sim_pair, columns=['from', 'to', 'sim']).to_csv(f'{PATH_OF_CODE}/sim_between_students.csv', index=False)
    return ls_sim_pair

def detect_sim_with_chatgpt(ls_student, ls_s_embedding, ls_gpt, ls_gpt_embedding):
    ls_sim_gpt = []
    for i in tqdm(range(len(ls_student)), desc='[Calculate ChatGPT Similarity]'):
        ls_tmp = []
        for j in range(len(ls_gpt)):
            ls_tmp.append(float(F.cosine_similarity(ls_s_embedding[i].flatten(), ls_gpt_embedding[j].flatten(), dim=-1).numpy()))
        ls_sim_gpt.append([f"{ls_student[i]['id']}_{ls_student[i]['name']}"] + ls_tmp)
    df_sim_gpt = pd.DataFrame(ls_sim_gpt, columns=['from'] + [f"sim_{gpt['name']}{gpt['id']}" for gpt in ls_gpt])
    df_sim_gpt['avg_sim'] = df_sim_gpt[[f"sim_{gpt['name']}{gpt['id']}" for gpt in ls_gpt]].mean(axis=1)
    df_sim_gpt.to_csv(f'{PATH_OF_CODE}/sim_with_chatgpt.csv', index=False)

    return ls_sim_gpt

def detect_outside_submit():
    return

def main():
    ls_student = preprocessing_student()
    ls_gpt = preprocessing_gpt()

    pretrained_name = ""
    if MODEL == 'codebert':
        pretrained_name = "microsoft/codebert-base" 
    elif MODEL == "unixcoder":
        pretrained_name = 'microsoft/unixcoder-base'
    elif MODEL == "codebert-py":
        pretrained_name = 'neulab/codebert-python'

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModel.from_pretrained(pretrained_name).to(DEVICE)

    ls_s_embedding = get_code_embedding(ls_student, model, tokenizer)
    ls_gpt_embedding = get_code_embedding(ls_gpt, model, tokenizer)

    ls_sim_pair = detect_sim_between_students(ls_student, ls_s_embedding)
    ls_sim_gpt = detect_sim_with_chatgpt(ls_student, ls_s_embedding, ls_gpt, ls_gpt_embedding)
    

if __name__ == "__main__":
    main()