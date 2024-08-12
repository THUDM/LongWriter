import json
import random
import requests
import multiprocessing
from tqdm import tqdm
import re

dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]
model = "LongWriter-glm4-9b"
filename = f"models/{model}/judge.jsonl"
prediction_file = open(f"models/{model}/pred.jsonl", "r", encoding="utf-8")

prompt_template = open("judge.txt", "r", encoding="utf-8").read()
fout = open(filename, 'w', encoding='utf-8')

GPT4_API_KEY = '' # Your API Key
GPT_MODEL = 'gpt-4o-2024-05-13'
def get_response_gpt4(prompt, temperature=0.5, max_new_tokens=1024, stop=None):
    tries = 0
    while tries < 10:
        tries += 1
        try:
            headers = {
                'Authorization': "Bearer {}".format(GPT4_API_KEY),
            }
            messages = [
                {'role': 'user', 'content': prompt},
            ]
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": GPT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop": stop,
            }, headers=headers, timeout=600)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
    else:
        print("Max tries. Failed.")
        return "Max tries. Failed."
    try:
        return resp["choices"][0]["message"]["content"]
    except: 
        return ''

def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def process_data(items):
    for item in tqdm(items):
        prompt = prompt_template.replace('$INST$', item['prompt']).replace('$RESPONSE$', item["response"])
        scores = None
        trys = 0
        while scores is None and trys < 5:
            output = get_response_gpt4(prompt)
            try:
                if '```json' in output:
                    output = extract_info(r'```json\n(.*?)\n```', output)
                output = output.replace('\n', '')
                scores = json.loads(output)
                for dim in dims:
                    if dim not in scores:
                        scores = None
                        trys += 1
            except Exception as e:
                trys += 1
        if scores is None:
            print(output)
        else:
            item['scores'] = scores
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')
            fout.flush()

data = [json.loads(line) for line in prediction_file]
random.shuffle(data)
PROC_NUM = 8
pool = multiprocessing.Pool(processes=PROC_NUM)
total = len(data)

for i in range(PROC_NUM):
    start = (i * total) // PROC_NUM
    end = None if i == PROC_NUM - 1 else ((i + 1) * total) // PROC_NUM
    pool.apply_async(process_data, args=(data[start:end],))

pool.close()
pool.join()
fout.close()

all_scores = [json.loads(line)['scores'] for line in open(filename, 'r', encoding='utf-8')]

total_score = dict()
for dim in dims:
    scores = [float(score[dim]) if dim in score else 3 for score in all_scores]
    total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
total_score['total'] = sum(total_score.values()) / len(total_score)
print(total_score)
with open(filename, 'a', encoding='utf-8') as fout:
    fout.write(json.dumps(total_score, ensure_ascii=False)+'\n')
