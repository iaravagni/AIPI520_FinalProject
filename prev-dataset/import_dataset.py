from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# # fetch dataset 
# cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

# # cdc_diabetes_health_indicators.data.to_csv('dataset.csv')
  
# # data (as pandas dataframes) 
# X = cdc_diabetes_health_indicators.data.features 
# y = cdc_diabetes_health_indicators.data.targets 

# # Merge the features and target dataframes if needed
# df = pd.concat([X, y], axis=1)

# df.to_csv('diabetes.csv')


# fetch dataset 
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
  
# data (as pandas dataframes) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

# # Merge the features and target dataframes if needed
df = pd.concat([X, y], axis=1)

df.to_csv('obesity.csv')