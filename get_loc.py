import pandas as pd
from PyGeoCN.regeo import regeo

# 输入和输出文件路径
input_file = 'data/lonlat.xlsx'  # 请替换为你的 Excel 文件路径
output_file = 'data/output.csv'  # 保存结果的 CSV 文件路径

# 读取 Excel 文件
df = pd.read_excel(input_file)

# 确保输入数据包含 'lon' 和 'lat' 列
if 'lon' not in df.columns or 'lat' not in df.columns:
    raise ValueError("Excel 文件中需要包含 'lon' 和 'lat' 列。")

# 添加新的列，存储省份和城市
df['Province'] = None
df['City'] = None
df['District'] = None

# 遍历每一行，调用 regeo 获取省、市、县信息
for index, row in df.iterrows():
    latitude = row['lat']
    longitude = row['lon']
    
    try:
        # 调用 regeo 获取地址信息
        result = regeo(latitude, longitude)
        
        if result['status'] == 1:
            # 提取省、市、县信息
            address = result.get('address', {})
            province = address.get('province', 'undefined')
            city = address.get('city', 'undefined')
            district = address.get('district', 'undefined')
            
            # 更新 DataFrame
            df.at[index, 'Province'] = province
            df.at[index, 'City'] = city
            df.at[index, 'District'] = district
        else:
            # 如果状态不是成功，记录 undefined
            df.at[index, 'Province'] = 'undefined'
            df.at[index, 'City'] = 'undefined'
            df.at[index, 'District'] = 'undefined'
    except Exception as e:
        # 捕获异常并记录 undefined
        print(f"Error processing row {index}: {e}")
        df.at[index, 'Province'] = 'undefined'
        df.at[index, 'City'] = 'undefined'
        df.at[index, 'District'] = 'undefined'

# 保存为 CSV 文件
df.to_csv(output_file, index=False)

print(f"结果已保存到 {output_file}")
