import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

class RandomTree:

    def __init__(self, data:str) -> None:
        self.dataframe = None
        self.import_data(data)
        self.subset_value = round(pow(self.dataframe.shape[1], 0.5))
        self.get_dataset_columns()

    def get_recommended_subset(self, subset_value):
        self.subset_value = subset_value
        print(f'Recommended subset size is: {subset_value}')
        return subset_value


    '''Importiert den Datensatz anhand des instanzierten Namens'''     

    def import_data(self, data):
        self.dataframe = pd.read_csv(f'{data}.csv')
        self.dataframe.dropna(inplace= True)
        return self.dataframe

    '''Löscht eine Spalte anhand des gegebenen Indexes'''

    def drop_column(self, column_index):
         self.dataframe.drop(self.dataframe.columns[column_index], axis=1, inplace=True)
         return self.get_dataset_columns()
    
    ''' Gibt die Namen aller Spalten des Datensatzes aus'''

    def get_dataset_columns(self):
        columns_list = list(self.dataframe.columns)
        print(f'Available Datapoints: {columns_list}')
        return columns_list

    '''Gibt die Werte einer Spalte aus'''            

    def get_data_from_column(self, column_index: int):
        column_data = self.dataframe[self.dataframe.columns[column_index]]
        print(column_data)
        return column_data

    '''Die Funtion teilt den DataFrame in Trainings- und Testdatensätze auf, 
       wobei 'parameter' die Features ohne die Zielvariable sind und `self.dataframe.columns[target]` 
       auf den Zielspaltenindex verweist. Sie erstellt stratifizierte Datensätze, so dass die proportionale
       Verteilung der Zielklassen erhalten bleibt. Sie gibt die aufgeteilten Trainings- und Testdatensätze 
       für Features (X) und die Zielvariable (y) zurück. '''

    def generate_stratified_test_sets(self, test_size: float, target: int):

        parameter = list(self.dataframe.drop(self.dataframe.columns[target], axis=1).columns)
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataframe[parameter],
            self.dataframe[self.dataframe.columns[target]],
            test_size=test_size,
            stratify=self.dataframe[self.dataframe.columns[target]])

        return X_train, X_test, y_train, y_test