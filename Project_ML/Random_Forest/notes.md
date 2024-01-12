
    predictions = {}
    for i, model in enumerate(models):
        predictions[f'model{i+1}'] = model.predict(data.df_future)


    plt.figure(figsize=(10, 5))

    for name, pred in predictions.items():
        plt.plot(data.df_future['Year'], pred, label=name)

    plt.xlabel('Jahre')
    plt.ylabel('CO2 Emissionen')
    plt.title('Vorhersage der CO2 Emissionen')
    plt.legend()

# Speichern des Diagramms im vorgegebenen Pfad.
    plt.savefig(r'C:\Machine Learning\Jup-Test\diagrams\predictions.png')





Future:data

def generate_future_data(self, end_year):
        df = self.dataframe
        
    
        target_name = df.columns[self.target_index]  
        year_index = df.columns.get_loc('Year')
        country_index = df.columns.get_loc('Country')
        energy_type_index = df.columns.get_loc('Energy_type')  
    
        start_year = df['Year'].max() + 1
        future_years_df = pd.DataFrame({'Year': range(start_year, end_year + 1)})
        future_data = pd.DataFrame()

        excluded_features = ['Year', 'Country', 'Energy_type']  
    
        for feature in df.columns.drop(target_name):
            if feature not in excluded_features:  
                X = df[['Year']].values
                y = df[feature].values
            
                model = LinearRegression()
                model.fit(X, y)
            
                future_values = model.predict(future_years_df[['Year']].values)
                future_data[feature] = future_values

    
        future_data.insert(country_index,"Country", value=195)
        future_data.insert(energy_type_index,'Energy_type', value=0)
        future_data.insert(year_index, 'Year', future_years_df['Year'])
        
        df_columns_without_target = df.columns.tolist()
        df_columns_without_target.remove(target_name)

        self.df_future = future_data
        return future_data