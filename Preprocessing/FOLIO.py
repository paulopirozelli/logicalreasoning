import pandas as pd
import os
import os.path as path

def loading(separator):
    # Parent directory two levels above
    parent_directory = path.abspath(path.join(__file__ ,"../.."))

    # Open datasets
    df_train = pd.read_csv(os.path.join(parent_directory, 'LogicData', 'FOLIO', 'folio-train.txt'), sep="	")
    df_test = pd.read_csv(os.path.join(parent_directory, 'LogicData', 'FOLIO', 'folio-validation.txt'), sep="	")

    # Merging and cleaning premises
    df_train['premises'] = df_train['premises'].str.replace('[\'', '', regex=False)
    df_train['premises'] = df_train['premises'].str.replace('\']', '', regex=False)
    df_train['premises'] = df_train['premises'].str.replace('\', \'', ' ', regex=False)
    df_train['premises'] = df_train['premises'].str.replace('[\"', '', regex=False)
    df_train['premises'] = df_train['premises'].str.replace('\", \'', '', regex=False)

    df_test['premises'] = df_test['premises'].str.replace('[\'', '', regex=False)
    df_test['premises'] = df_test['premises'].str.replace('\']', '', regex=False)
    df_test['premises'] = df_test['premises'].str.replace('\', \'', ' ', regex=False)
    df_test['premises'] = df_test['premises'].str.replace('[\"', '', regex=False)
    df_test['premises'] = df_test['premises'].str.replace('\", \'', '', regex=False)

    # Third label was different in the validation set
    df_test['label'] = df_test['label'].str.replace('Uncertain', 'Unknown', regex=False)

    # Create context
    df_train['text'] = df_train['premises'] + separator + df_train['conclusion']
    df_test['text'] = df_test['premises'] + separator + df_test['conclusion']

    # Replace Labels by numbers
    # False = 0, True = 1, Unknown = 2
    df_train['label'] = df_train['label'].replace({'False': 0, 'True': 1, 'Unknown': 2})
    df_test['label'] = df_test['label'].replace({'False': 0, 'True': 1, 'Unknown': 2})

    # Select columns
    df_train = df_train[['text', 'label']]
    df_test = df_test[['text', 'label']]

    # Spliting into train, validaton, test sets
    # Dividing test set into Validation/Test
    # FOLIO repository does not provide validation split so we make test and validation the same
    df_validation = df_test

    return df_train, df_validation, df_test
