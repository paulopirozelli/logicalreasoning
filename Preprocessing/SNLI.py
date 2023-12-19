import pandas as pd
import os
import os.path as path
from sklearn.model_selection import train_test_split

parent_directory = path.abspath(path.join(__file__ ,"../.."))
folder = os.path.join('Data', 'NLI', 'SNLI')

def loading(separator):
    dataset_list = []

    for split in ['snli_1.0_train.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt']:

        file_path = os.path.join(parent_directory, folder, split)

        df = pd.read_csv(file_path, sep="	")

        # Rename colum
        df = df.rename(columns={'gold_label': 'label'})

        # Eliminate observations without label
        df = df[df['label'].isin(['contradiction', 'entailment', 'neutral'])]

        # Create context
        df['text'] = df['sentence1'] + separator + df['sentence2']

        # Convert text to str
        df['text'] = df['text'].astype('str')
        
        # Replace Labels by numbers
        df['label'] = df['label'].replace({'contradiction': 0, 'entailment': 1, 'neutral': 2})

        # Select specific columns
        df = df[['text', 'label']]

        dataset_list.append(df)

    return dataset_list[0], dataset_list[1], dataset_list[2]









