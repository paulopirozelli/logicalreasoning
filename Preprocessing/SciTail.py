import pandas as pd

from datasets import load_dataset

dataset = load_dataset("bigscience-biomedical/scitail")

def loading(separator):
    dataset_list = []

    for split in ['train', 'validation', 'test']:
        df = pd.DataFrame(dataset[split])

        # Create context
        df['text'] = df['premise'] + separator + df['hypothesis']

        # Select specific columns
        df = df[['text', 'label']]

        # Replace Labels by numbers
        # False = 0, True = 1, Unknown = 2
        # entails = 1, neutral = 0

        df['label'] = df['label'].replace({'entails': 1, 'neutral': 0})

        dataset_list.append(df)

    return dataset_list[0], dataset_list[1], dataset_list[2]