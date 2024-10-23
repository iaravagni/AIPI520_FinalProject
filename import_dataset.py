from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

# cdc_diabetes_health_indicators.data.to_csv('dataset.csv')
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 

# Merge the features and target dataframes if needed
df = pd.concat([X, y], axis=1)

df.to_csv('diabetes.csv')
