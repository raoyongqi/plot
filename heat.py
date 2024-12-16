import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 Excel 文件
df = pd.read_excel('data/climate_soil_tif.xlsx')
print(df.columns)
# 设置行和列的名称
rows = ['LON', 'wc2.1_5m_srad_04', 'wc2.1_5m_srad_07', 'wc2.1_5m_srad_02']
cols = ['LON', 'wc2.1_5m_srad_04', 'wc2.1_5m_srad_07', 'wc2.1_5m_srad_02']
# 将列表中的元素转为大写
rows = [item.upper() for item in rows]
cols = [item.upper() for item in cols]

columns_to_select = rows
df_selected = df[columns_to_select]

# 计算相关系数矩阵
corr_matrix = df_selected.corr()

# 创建掩码，只显示下三角部分
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  # k=0 包括对角线

# 提取下三角部分的相关系数（不包括对角线）
lower_triangle_corr = corr_matrix.where(~mask)

# 过滤掉全为 NaN 的行和列
heatmap_data = lower_triangle_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

# 绘制热图
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)


# 显示数值
for (i, j), val in np.ndenumerate(heatmap_data):
    if not np.isnan(val):  # 只在有值的地方显示
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black',fontsize=14)

# 设置刻度
# 设置刻度
plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45, fontsize=14)  # 设置 x 轴刻度字体大小
plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index, fontsize=14)  # 设置 y 轴刻度字体大小


title = 'Correlation Matrix of Selected Variables (Including Diagonal)'
plt.title(title, fontsize=16)  # 设置标题字体大小

title ='Correlation Matrix of Selected Variables (Including Diagonal)'
# 设置标题
output_file_path = f'data/{title}.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

# 显示热图
# 显示热图
plt.tight_layout()  # 调整布局
plt.show()
