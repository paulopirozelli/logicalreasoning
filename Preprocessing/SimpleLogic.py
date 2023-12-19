import os
import os.path as path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Example file
example = 0
example_name = 'prop_examples_' + str(example) + '.txt'

# Open JSON file
parent_directory = path.abspath(path.join(__file__, "../.."))
folder = os.path.join('LogicData', 'SimpleLogic')
file_path = os.path.join(parent_directory, folder, example_name)

# Open JSON file
with open(file_path, 'r') as f:
    json_list = json.load(f)

def loading(separator):
    # Extract Text (fact + rules) and Label

    text = []
    label = []

    for i in range(len(json_list)):

        # predicates
        predicate = '. Alice is '.join(json_list[i]['preds'])
        predicate = 'Alice is ' + predicate  # 'Alice is' in the first predicate
        predicate += '. '  # final dot

        # rules
        individual_rules = []

        for j in json_list[i]['rules']:
            antecedent = ' and '.join(j[0])
            consequent = j[1]
            rule = 'if Alice is ' + antecedent + ', then Alice is ' + consequent
            rule += '.'
            individual_rules.append(rule)

        rule_set = ' '.join(individual_rules)

        # context: predicates + rules
        context = predicate + rule_set

        # text: context + [SEP] + hypothesis
        for k in range(len(json_list[i]['facts'])):
            fact = 'Alice is ' + json_list[i]['facts'][k] + '.'
            texts = context + separator + fact
            text.append(texts)

            # label
            label.append(json_list[i]['label'])

    df = pd.DataFrame({'text': text, 'label': label})

    # Balance data
    df = df.groupby('label').sample(df.groupby('label').size().min(), random_state=1232)

    # Spliting into train, validaton, test sets
    # Train = 80%, Validation = 10%, Test = 10%
    df_test, df_train = train_test_split(df, test_size=0.8, random_state=42)
    df_test, df_validation = train_test_split(df_test, test_size=0.5, random_state=42)

    return df_train, df_validation, df_test
