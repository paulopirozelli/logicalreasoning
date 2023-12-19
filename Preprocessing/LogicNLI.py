import os
import os.path as path
import json
import pandas as pd

files = ['train_language.json', 'dev_language.json', 'test_language.json']
parent_directory = path.abspath(path.join(__file__, "../.."))

def loading(separator):
    data = []

    for name in files:

        folder = os.path.join('LogicData', 'LogicNLI')
        file_path = os.path.join(parent_directory, folder, name)

        # Open file
        with open(file_path, 'r') as f:
            json_list = json.load(f)

        # Create dataset
        text = []
        label = []

        for i in range(len(json_list)):

            # predicates
            predicate = ' '.join(json_list[str(i)]['facts'])

            # rules
            rule = ' '.join(json_list[str(i)]['rules'])

            # context
            context = predicate + ' ' + rule

            for j in range(len(json_list[str(i)]['statements'])):
                # select fact j
                fact = json_list[str(i)]['statements'][j]

                # concatenate with context
                txt = context + separator + fact
                text.append(txt)

                # select label
                labeling = json_list[str(i)]['labels'][j]
                label.append(labeling)

        df = pd.DataFrame({'text': text, 'label': label})

        df['label'] = df['label'].replace({'contradiction': 0, 'entailment': 1,
                                           'neutral': 2, 'self_contradiction': 3})

        data.append(df)

    return data[0], data[1], data[2]