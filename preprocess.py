import pandas as pd
import json
class MissingColumnError(AttributeError):
    """Error indicating that an imported DataFrame lacks necessary columns"""
    print (AttributeError)
    pass

def load_arguments(filepath):
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        if not {'Argument ID', 'Premise'}.issubset(set(dataframe.columns.values)):
            raise MissingColumnError('The argument "%s" file does not contain the minimum required columns [Argument ID, Premise].' % filepath)
        return dataframe
    except IOError:
        print("Error in load_arguments" + IOError.strerror)
        raise

def load_label(filepath, label_order):
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        dataframe = dataframe[['Argument ID'] + label_order]
        return dataframe
    except IOError:
        print("Error in load_label" + IOError.strerror)
        raise
    except KeyError:
        raise MissingColumnError('The file "%s" does not contain the required columns for its level.' % filepath)

def load_json_file(filepath):
    """Load content of json-file from `filepath`"""
    with open(filepath, 'r') as  json_file:
        return json.load(json_file)


def load_values_from_json(filepath):
    """Load values per level from json-file from `filepath`"""
    json_values = load_json_file(filepath)
    values = { "1":list() }
    for value in json_values:
        values["1"].append(value)
    values["1"] = sorted(values["1"])
    return values


def combine_columns(df_arguments, df_labels):
    return pd.merge(df_arguments, df_labels, on='Argument ID')


def split_arguments(df_arguments):
    train_arguments = df_arguments.sample(frac=0.8,random_state=200)
    test_arguments = df_arguments.drop(train_arguments.index)
    return train_arguments, test_arguments
