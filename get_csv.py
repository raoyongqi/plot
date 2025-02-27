import pandas as pd

# Creating the dataframe based on the provided data
data = {
    'Response': ['Plant Disease'] * 17,
    'Feature': ['SRAD', 'SAND', 'TSEA', 'LON', 'VAPR', 'MAX MAT', 'WIND', 'PSEA', 'LAT', 'ELEV', 'MAP', 'AVG MAT', 
                'MIN MAT', 'S CLAY', 'T SAND', 'T REF BULK', 'T BULK DEN'],
    'Category': ['Climate', 'Soil', 'Climate', 'Geography', 'Climate', 'Climate', 'Climate', 'Climate', 'Geography', 
                 'Geography', 'Climate', 'Climate', 'Climate', 'Soil', 'Soil', 'Soil', 'Soil'],
    'Importance': [0.15, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02]
}

# Create DataFrame
df = pd.DataFrame(data)

# Saving the dataframe to an Excel file
file_path = 'plant_disease_features.xlsx'
df.to_excel(file_path, index=False)

file_path
