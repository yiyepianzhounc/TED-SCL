import os
import openai
import pandas as pd
import random
from standard_datasets.unifiedDatasetsLoader import unifiedDatasets
import time
import dashscope
import json
import requests
import concurrent.futures

# 替换为你的实际 Key
openai.api_key = '**'
dashscope.api_key = '**'

# 代理列表（示例）
proxy_list = ['**'] * 36

# 设置 HTTP 代理
os.environ['HTTP_PROXY'] = 'http://' + proxy_list[5]
os.environ['HTTPS_PROXY'] = 'http://' + proxy_list[5]


def create_prompt(sentence, euphemism, explanation):
    prompt3 = "Please determine whether the following sentence is a euphemistic expression. Output 1 for a euphemistic sentence and 0 for a non-euphemistic sentence."
    user_Input = "Sentence: {sentence}, Euphemistic Words: {euphemism}, Euphemistic Meanings: {explanation}".format(
        sentence=sentence, euphemism=euphemism, explanation=explanation
    )
    return prompt3, user_Input


def OpenAI_answer(instructions, texts, model_name, proxy):
    generation_config = dict(
        max_tokens=2,
        timeout=5,
        temperature=0,
        top_p=0.01
    )
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": texts}
            ],
            max_tokens=generation_config["max_tokens"],
            timeout=generation_config["timeout"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"]
        )
        return response['choices'][0]['message']['content']
    except Exception:
        return 'error'


def Aliyun_answer(instructions, texts, model_name, proxy):
    messages = [
        {'role':'system','content':instructions},
        {'role':'user','content': texts}
    ]
    responses = dashscope.Generation.call(
        model=str(model_name),
        messages=messages,
        result_format='message',
        stream=False,
        incremental_output=False
    )
    try:
        return responses.output.choices[0]['message']['content']
    except Exception:
        return 'error'


def model_name_Transfer(model_choice):
    mapping = {
        'gpt35': 'gpt-3.5',
        'gpt4o': 'gpt-4o',
        'gpt4t': 'gpt-4-turbo',
        'gpt35t': 'gpt-3.5-turbo'
    }
    return mapping.get(model_choice, model_choice)


def model_name_Transfer_Aliyun(model_choice):
    mapping = {
        'qwen2_7b': 'qwen2-7b-instruct',
        'qwen2_72b': 'qwen2-72b-instruct',
        'llama3_8b': 'llama3-8b-instruct',
        'llama3_70b': 'llama3-70b-instruct',
        'chatglm3': 'chatglm-6b-v2'
    }
    return mapping.get(model_choice, model_choice)


def API_Label_by_datasets_OpenAI(datasets_name, model_choice):
    LLM_name = model_name_Transfer(model_choice)
    dataset_loader = unifiedDatasets(datasets_name)
    all_data = dataset_loader.load_test()

    result_df = pd.DataFrame(columns=['sentence', 'is_euph', 'keyword', 'category', 'model_output'])
    output_dir = f"./labeling/{datasets_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{LLM_name}.csv")

    if os.path.exists(output_file):
        result_df = pd.read_csv(output_file, encoding='gbk')
    start_index = len(result_df)

    for index, row in all_data.iloc[start_index:].iterrows():
        sentence = row['sentence']
        keyword = row['keyword']
        category = row['category']
        is_euph = row['is_euph']

        prompt, user_input = create_prompt(sentence, keyword, category)
        model_output = OpenAI_answer(prompt, user_input, LLM_name, proxy_list[random.randint(0, len(proxy_list)-1)])
        model_output = 1 if '1' in str(model_output) else 0

        time.sleep(1)
        temp_df = pd.DataFrame([{
            'sentence': sentence,
            'is_euph': is_euph,
            'keyword': keyword,
            'category': category,
            'model_output': model_output
        }])
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

        if (index + 1) % 5 == 0:
            result_df.to_csv(output_file, index=False, encoding='gbk')
            print(f"Saved results for {index + 1} rows.")

    result_df.to_csv(output_file, index=False, encoding='gbk')
    print(f"All results saved to {output_file}.")


def API_Label_by_datasets_Aliyun(datasets_name, model_choice):
    LLM_name = model_name_Transfer_Aliyun(model_choice)
    dataset_loader = unifiedDatasets(datasets_name)
    all_data = dataset_loader.load_test()

    result_df = pd.DataFrame(columns=['sentence', 'is_euph', 'keyword', 'category', 'model_output'])
    output_dir = f"./labeling/{datasets_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{LLM_name}.csv")

    if os.path.exists(output_file):
        result_df = pd.read_csv(output_file, encoding='gbk')
    start_index = len(result_df)

    for index, row in all_data.iloc[start_index:].iterrows():
        sentence = row['sentence']
        keyword = row['keyword']
        category = row['category']
        is_euph = row['is_euph']

        prompt, user_input = create_prompt(sentence, keyword, category)
        model_output = Aliyun_answer(prompt, user_input, LLM_name, proxy_list[random.randint(0, len(proxy_list)-1)])
        model_output = 1 if '1' in str(model_output) else 0

        time.sleep(3)
        temp_df = pd.DataFrame([{
            'sentence': sentence,
            'is_euph': is_euph,
            'keyword': keyword,
            'category': category,
            'model_output': model_output
        }])
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

        if (index + 1) % 5 == 0:
            result_df.to_csv(output_file, index=False, encoding='gbk')
            print(f"Saved results for {index + 1} rows.")

    result_df.to_csv(output_file, index=False, encoding='gbk')
    print(f"All results saved to {output_file}.")


def process_datasets_multithreaded_Aliyun(datasets_name, model_choice_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(API_Label_by_datasets_Aliyun, datasets_name, m)
            for m in model_choice_list
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


def process_datasets_multithreaded_OpenAI(datasets_name, model_choice_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(API_Label_by_datasets_OpenAI, datasets_name, m)
            for m in model_choice_list
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


if __name__ == '__main__':
    datasets_name_list = ['EACL_EN_24', 'FigLang_EN_24', 'JointEDI_EN_24']
    for datasets_name in datasets_name_list:
        model_choice_list = ['llama3_8b', 'llama3_70b']
        process_datasets_multithreaded_Aliyun(datasets_name, model_choice_list)
