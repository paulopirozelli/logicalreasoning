# Dataset version
# options: NatLang, depth-1, depth-2, depth-3, depth-3ext, depth-3NatLang, depth-4, depth-5, birds-eletricity
# birds-eletricity tem somente test set
dataset_version = 'NatLang'

import os
import os.path as path
import json
import pandas as pd

parent_directory = path.abspath(path.join(__file__, "../.."))
folder = os.path.join('LogicData', 'RuleTaker', 'problog')

def loading(separator):
    dataset_list = []

    for split in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:

        file_path = os.path.join(parent_directory, folder, dataset_version, split)

        with open(file_path, 'r') as json_file:
            json_list = list(json_file)

        full_list = []

        for json_str in json_list:
            result = json.loads(json_str)
            full_list.append(result)

        label = []
        text = []

        for i in range(len(full_list)):
            label.append(full_list[i]['theory_assertion_instance']['label'])

            assertion = full_list[i]['english']['assertion_statement']

            theory_list = full_list[i]['english']['theory_statements']
            context = ''
            for j in range(len(theory_list)):
                context += theory_list[j] + ' '
            theory = context[:-1]

            ctx = theory + separator + assertion
            text.append(ctx)

        df = pd.DataFrame({'text': text, 'label': label})
        df['label'] = df['label'].replace({False: 0, True: 1})
        dataset_list.append(df)

    return dataset_list[0], dataset_list[1], dataset_list[2]