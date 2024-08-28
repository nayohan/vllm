import pandas as pd
from datasets import load_dataset

dataset_name = 'nayohan/Magpie-Pro-MT-300K-v0.1-ko'
dataset_name = 'werty1248/sharegpt-tagengo-gpt4-ko'
dataset = load_dataset(dataset_name, cache_dir="/data/project/yohan/99_dataset", split='train')

def convert_to_multi_turn_conversation(example):
    chat = {
        "conversations": [
            {"from": "human", "value": example['input1']},
            {"from": "gpt", "value": example['output1']},
            {"from": "human", "value": example['input2']},
            {"from": "gpt", "value": example['output2']},
        ]
    }
    return chat

# dataset = dataset.map(convert_to_multi_turn_conversation, num_proc=128)
df = pd.DataFrame(dataset)#.sample(2000, random_state=22)
output_name = dataset_name.split('/')[-1]
df.to_json(f'{output_name}.json', orient='records', force_ascii=False)