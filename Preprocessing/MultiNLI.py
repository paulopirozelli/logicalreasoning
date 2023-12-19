import pandas as pd
import os
import os.path as path
from sklearn.model_selection import train_test_split

# There are matched dev/test sets which are derived from the same sources as those in the training set,
# and mismatched sets which do not closely resemble any genre seen at training time.
dataset_version = 'matched'
dataset_name = 'multinli_1.0_dev_' + dataset_version + '.txt'

parent_directory = path.abspath(path.join(__file__, "../.."))
folder = os.path.join('Data', 'NLI', 'MultiNLI')

def loading(separator):
    dataset_list = []

    for split in ['multinli_1.0_train.txt', dataset_name]:
        file_path = os.path.join(parent_directory, folder, split)

        df = pd.read_csv(file_path, sep="	", on_bad_lines='skip')

        # Rename colum
        df = df.rename(columns={'gold_label': 'label'})

        # Eliminate observations without label
        df = df[df['label'].isin(['contradiction', 'entailment', 'neutral'])]

        # Create context
        df['text'] = df['sentence1'] + separator + df['sentence2']

        # Convert text to str and label to int
        df['text'] = df['text'].astype('str')

        # Replace Labels by numbers
        df['label'] = df['label'].replace({'contradiction': 0, 'entailment': 1, 'neutral': 2})

        # Select specific columns
        df = df[['text', 'label']]

        dataset_list.append(df)

    df_validation, df_test = train_test_split(dataset_list[1], test_size=0.5)
    return dataset_list[0], df_validation, df_test









