import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

class TrainingRandomTree:
        
        def __init__(self) -> None:
                pass

def generate_stratified_test_sets(self, test_size: float, target: int):

        parameter = list(self.dataframe.drop(self.dataframe.columns[target], axis=1).columns)
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataframe[parameter],
            self.dataframe[self.dataframe.columns[target]],
            test_size=test_size,
            stratify=self.dataframe[self.dataframe.columns[target]])

        return X_train, X_test, y_train, y_test