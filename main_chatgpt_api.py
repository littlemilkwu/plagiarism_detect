import os
import numpy as np
import pandas as pd
import openai
import zipfile
import re
from pprint import pprint
from tqdm import tqdm
from glob import glob
from pprint import pprint
from config import OPEN_API_KEY
from argparse import ArgumentParser
openai.api_key = OPEN_API_KEY

parser = ArgumentParser()
parser.add_argument('--code_path', '-c', default='./1112_H210301-PE5-20230426', type=str)
args = parser.parse_args()
CODE_PATH = args.code_path
QUESTION = os.path.join(CODE_PATH, 'question.txt')
GPT_TIMES = 3

def get_py_txt(py_path):
    f = open(py_path)
    ls_lines = f.readlines()
    str_py = ''
    for line in ls_lines:
        if '#' in line:
            line = line.split('#')[0]
        if len(line.strip()) != 0:
            str_py = str_py + line
    f.close()
    return str_py

def preprocessing_student():
    ls_path = [item for item in os.listdir(CODE_PATH) if os.path.isdir(os.path.join(CODE_PATH, item))]
    ls_student = []
    # for i, path in enumerate(ls_path):
    #     print(i, path)
    for path in ls_path:
        s_id, s_name, _ = path.split('_')
        s_name = s_name.strip().replace('　', '').replace(' ', '')
        ls_zip =  glob(os.path.join(CODE_PATH, path, '*.zip'))
        if len(ls_zip) != 0:
            zip = zipfile.ZipFile(ls_zip[0])
            zip.extractall(os.path.dirname(ls_zip[0]))
            zip.close()

        ls_py = glob(os.path.join(CODE_PATH, path, '*.py')) + glob(os.path.join(CODE_PATH, path, '**/*.py'))
        if len(ls_py) == 0:
            print(f"{s_name} 沒有 .py 檔案")
            continue
        elif len(ls_py) > 1:
            print(f"{s_name} 有多個 .py 檔案，以 {ls_py[0]}為主")
        s_py = get_py_txt(ls_py[0])
        ls_student.append(dict(id=s_id, name=s_name, py_txt=s_py))
    return ls_student


def ask_chatgpt_sim(question, py_code, verbose=0):
    background = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\n"
    verbose = "僅回傳相似度數值，不用任何說明，15 字以內回答。" if verbose == 0 else ''
    query = f'根據以下的題目說明\n「{question}」\n' + \
        f'先自行思考 ChatGPT 你可能提供的 Python 解答方案，再從邏輯和解題策略幫我判斷下方的<Python 程式碼>的與 ChatGPT 你可能提供的 Python 解答方案的相似度，' + \
        f'只需給我一個介於 0 至 1 之間的相似度數值（浮點數，精度至小數點後 2 位），0 代表完全不相似，1代表完全地相似。\n' + \
        f'<Python 程式碼>：\n{py_code}\n\n'+\
        f'{verbose}'
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.7,
        messages=[
            # {"role": "system", "content": background},
            {"role": "user", "content": background + query}
        ]
    )
    return response

def main():
    ls_student = preprocessing_student()
    f = open(QUESTION)
    str_question = f.read()
    f.close()
    # pprint(ls_student[10:13])
    # print(str_question)
    ls_csv = []
    for student in tqdm(ls_student, desc='[Asking ChatGPT (Spending Money QQ)]'):
        ls_raw = []
        for i in range(GPT_TIMES):
            gpt_response = ask_chatgpt_sim(str_question, student['py_txt'])
            gpt_raw_ans = gpt_response['choices'][0]['message']['content']
            gpt_raw_ans = gpt_raw_ans.replace('\n', ' ')
            ls_raw.append(gpt_raw_ans)

        ls_re = []
        for raw_ans in ls_raw:
            pattern = r'\d+\.\d+|\d'
            re_result = re.findall(pattern, raw_ans)
            if len(re_result) == 0:
                print(f'沒找到 pattern from: "{raw_ans}"')
                ls_re.append(-1.0)
                continue
            elif len(re_result) > 1:
                print(f'有多個 pattern from: "{raw_ans}"')
            ls_re.append(float(re_result[0]))

        ls_csv.append([student['id'], student['name']] + ls_raw + ls_re)
    df_csv = pd.DataFrame(ls_csv, columns=['id', 'name'] + [f'gpt_raw_{i+1}' for i in range(GPT_TIMES)] + [f'gpt_re_{i+1}' for i in range(GPT_TIMES)])
    df_csv['avg_gpt'] = df_csv[[f'gpt_re_{i+1}' for i in range(GPT_TIMES)]].mean(axis=1)
    df_csv['max_gpt'] = df_csv[[f'gpt_re_{i+1}' for i in range(GPT_TIMES)]].max(axis=1)
    df_csv['min_gpt'] = df_csv[[f'gpt_re_{i+1}' for i in range(GPT_TIMES)]].min(axis=1)
    df_csv.to_csv(os.path.join(CODE_PATH, 'sim_ans_by_chatgpt.csv'), index=False)

if __name__ == "__main__":
    main()