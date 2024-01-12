import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DatasetImport:

    def __init__(self, data:str, target_index) -> None:
        self.target_index = target_index
        self.dataframe = None
        self.import_data(data)
        self.subset_value = round(pow(self.dataframe.shape[1], 0.5))
        self.drop_column(0)
        self.get_dataset_columns()
        self.label_encoding(0)	
        self.label_encoding(1)
        

    

    def get_recommended_subset(self, subset_value):
        self.subset_value = subset_value
        print(f'Recommended subset size is: {subset_value}')
        return subset_value   

    def import_data(self, data):
        self.dataframe = pd.read_csv(f'{data}.csv')
        self.dataframe.dropna(inplace= True)
        return self.dataframe

    def drop_column(self, column_index):
         self.dataframe.drop(self.dataframe.columns[column_index], axis=1, inplace=True)
         return self.get_dataset_columns()

    def get_dataset_columns(self):
        columns_list = list(self.dataframe.columns)
        #print(f'Available Datapoints: {columns_list}')
        return columns_list          

    def get_data_from_column(self, column_index: int):
        column_data = self.dataframe[self.dataframe.columns[column_index]]
        #print(column_data)
        return column_data
    
    def label_encoding(self, column_index: int):
        label_encoder = LabelEncoder()
        self.dataframe[self.dataframe.columns[column_index]] = label_encoder.fit_transform(
            self.dataframe[self.dataframe.columns[column_index]]
        )
    
    def generate_test_sets(self, test_size: float):
        parameter = list(self.dataframe.drop(self.dataframe.columns[self.target_index], axis=1).columns)
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataframe[parameter],
            self.dataframe[self.dataframe.columns[self.target_index]],
            test_size=test_size
        )
        return X_train, X_test, y_train, y_test
    
    def generate_future_data(self, end_year):
    # Filtern des DataFrames
        df_filtered = self.dataframe[
        (self.dataframe['Country'] != 195) &
        (self.dataframe['Energy_type'] != 0) &
        (self.dataframe['Year'] >= 2005)
    ]

    # Festlegen des niedrigsten und höchsten Ländercodes
        min_country_code = df_filtered['Country'].min()
        max_country_code = df_filtered['Country'].max()

        future_data_list = []
        energy_type_list = [1, 2, 3, 4, 5]
    
    # Wir gehen davon aus, dass 'target_index' für die Zielvariable bekannt ist.
        target_name = self.dataframe.columns[self.target_index]
        excluded_features = ['Year', 'Country', 'Energy_type', 'CO2_emission']

    # Regression für jedes Land
        for country_code in range(min_country_code, max_country_code + 1):
            df_country = df_filtered[df_filtered['Country'] == country_code]
        
            if df_country.empty:
                continue  # Überspringen, wenn keine Daten für den Ländercode vorhanden sind

            start_year = df_country['Year'].max() + 1
            future_years = range(start_year, end_year + 1)
            future_data = pd.DataFrame({'Year': list(future_years)})

            # Durchlaufe alle Features außer ausgeschlossene
            for feature in set(df_country.columns) - set(excluded_features):
                if df_country[feature].notna().all():  # Überspringe Feature, wenn NaN-Werte vorhanden sind
                    X = df_country[['Year']]
                    y = df_country[feature]
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(future_data[['Year']])
                    future_data[feature] = abs(predictions)
        
        # Nehme den meistgenutzten Energy_type des jeweiligen Landes, der nicht 0 ist
            for energy_type in energy_type_list:
                future_data_type = future_data.copy()
                future_data_type['Country'] = country_code
                future_data_type['Energy_type'] = energy_type
                future_data_list.append(future_data_type)

    # Kombiniere alle DataFrames aus der Liste in ein einziges DataFrame
        combined_future_data = pd.concat(future_data_list, ignore_index=True)
    
    # Bereinige das DataFrame
        combined_future_data.dropna(inplace=True)  # Entfernt Zeilen mit NaN-Werten
        desired_order = [
    'Country', 'Energy_type', 'Year', 'Energy_consumption', 'Energy_production',
    'GDP', 'Population', 'Energy_intensity_per_capita', 'Energy_intensity_by_GDP']          
        combined_future_data = combined_future_data[desired_order]
    
    # Speichere das kombinierte DataFrame in der Klasse für zukünftige Verwendung
        self.df_future = combined_future_data
        return combined_future_data

   
    
if __name__ == '__main__':
    
    data = DatasetImport('energy', 9)
    data.generate_future_data(2030)
    print(data.df_future)

    model_paths = [
    'Project_ML\\Random_Forest\\models_2\\best_model_estimators_test_size4_200_rs11.pkl',
    'Project_ML\\Random_Forest\\models_2\\best_model_estimators_300_rs8.pkl',
    'Project_ML\\Random_Forest\\models_2\\best_model_estimators_test_size4_150_rs5.pkl',
    'Project_ML\\Random_Forest\\models_2\\best_model_estimators_test_size4_325_rs1.pkl',
]
    models = []

    for model_path in model_paths:
        with open(model_path, 'rb') as file:
            models.append(pickle.load(file))

    historical_filtered = data.dataframe[
    (data.dataframe['Country'] == 195) &
    (data.dataframe['Energy_type'] == 0) &
    (data.dataframe['Year'] >= 1981)
]
    historical_co2_emissions = historical_filtered[['Year', 'CO2_emission']]

    predictions_future_df = pd.DataFrame({
    'Year': data.df_future['Year'].unique(),
    'CO2_emission_sum': 0
})

for i, model in enumerate(models):
    prediction = model.predict(data.df_future)
    temp_df = pd.DataFrame({
    'Year': data.df_future['Year'],
    'CO2_emission': prediction
    })
    
    summed_emissions_by_year = temp_df.groupby('Year', as_index=False)['CO2_emission'].sum()
    predictions_future_df = predictions_future_df.merge(summed_emissions_by_year,
                                                       on='Year', how='left')
    predictions_future_df['CO2_emission_sum'] += predictions_future_df['CO2_emission'].fillna(0)
    predictions_future_df.drop(columns='CO2_emission', inplace=True)
    predictions_future_df = predictions_future_df[predictions_future_df['Year'] != 2019]
    predictions_future_df.sort_values(by='Year', inplace=True) 
    print(predictions_future_df)
    plt.figure(figsize=(10, 5))
    plt.plot(historical_co2_emissions['Year'], historical_co2_emissions['CO2_emission'], label='Historische Daten', color='red')
    plt.plot(predictions_future_df['Year'], predictions_future_df['CO2_emission_sum'], label=f'Modell {i+1} Vorhersage', color='green')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissionen in Megatonnen')
    plt.title(f'CO2 Emissions Voraussage - Modell {i+1}')
    plt.legend()
    plt.savefig(f'C:\\Machine Learning\\machine_learning_hm_new\\machine_learning_hm\\Project_ML\\Random_Forest\\new_figures\\predictions_model{i+1}.png')
    plt.close()
    predictions_future_df['CO2_emission_sum'] = 0






    


