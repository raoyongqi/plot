import pandas as pd
from PyGeoCN.regeo import regeo

input_file = 'data/climate_soil_tif.xlsx'
output_file = 'data/climate_soil_loc.csv'

df = pd.read_excel(input_file)
df.columns = df.columns.str.lower()
if 'lon' not in df.columns or 'lat' not in df.columns:
    raise ValueError("Excel 文件中需要包含 'lon' 和 'lat' 列。")

df['Province'] = None
df['City'] = None
df['District'] = None

for index, row in df.iterrows():
    latitude = row['lat']
    longitude = row['lon']
    
    try:
        result = regeo(latitude, longitude)
        
        if result['status'] == 1:

            address = result.get('address', {})
            province = address.get('province', 'undefined')
            city = address.get('city', 'undefined')
            district = address.get('district', 'undefined')
            
            df.at[index, 'Province'] = province
            df.at[index, 'City'] = city
            df.at[index, 'District'] = district
        else:

            df.at[index, 'Province'] = 'undefined'
            df.at[index, 'City'] = 'undefined'
            df.at[index, 'District'] = 'undefined'
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df.at[index, 'Province'] = 'undefined'
        df.at[index, 'City'] = 'undefined'
        df.at[index, 'District'] = 'undefined'

# 保存为 CSV 文件
df.to_csv(output_file, index=False)

print(f"结果已保存到 {output_file}")
