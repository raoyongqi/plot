import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data/climate_soil_tif.xlsx')
print(df.columns)

rows = ['LON', 'wc2.1_5m_srad_04', 'wc2.1_5m_srad_07', 'wc2.1_5m_srad_02']
cols = ['LON', 'wc2.1_5m_srad_04', 'wc2.1_5m_srad_07', 'wc2.1_5m_srad_02']

rows = [item.upper() for item in rows]
cols = [item.upper() for item in cols]

columns_to_select = rows
df_selected = df[columns_to_select]

corr_matrix = df_selected.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)

lower_triangle_corr = corr_matrix.where(~mask)

heatmap_data = lower_triangle_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)


for (i, j), val in np.ndenumerate(heatmap_data):
    if not np.isnan(val):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black',fontsize=14)

plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45, fontsize=14)
plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index, fontsize=14)


title = 'Correlation Matrix of Selected Variables (Including Diagonal)'
plt.title(title, fontsize=16)

title ='Correlation Matrix of Selected Variables (Including Diagonal)'


output_file_path = f'data/{title}.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()
