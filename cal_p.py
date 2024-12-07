import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rcParams

# 设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 显示中文
# rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 CSV 文件
def read_csv_file(file_path):
    return pd.read_csv(file_path)

# 统计数据中Site、Province、City、District的数量
def count_values(df):
    site_counter = Counter()
    province_counter = Counter()
    city_counter = Counter()
    district_counter = Counter()

    # 遍历数据框中的每一行
    for _, row in df.iterrows():
        site = row.get('Site')
        province = row.get('Province')
        city = row.get('City')
        district = row.get('District')

        if site:
            site_counter[site] += 1
        if province:
            province_counter[province] += 1
        if city:
            city_counter[city] += 1
        if district:
            district_counter[district] += 1

    # 按重复次数从高到低排序
    site_sorted = site_counter.most_common()
    province_sorted = province_counter.most_common()
    city_sorted = city_counter.most_common()
    district_sorted = district_counter.most_common()

    return site_sorted, province_sorted, city_sorted, district_sorted

# 绘制并保存条形图
def plot_and_save_bar_chart(data, title, filename):
    # 只取前10个数据
    top_10 = data[:10]
    labels, values = zip(*top_10)

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    
    # 调整 x 和 y 轴标签的字体大小
    plt.xlabel('Count', fontsize=22)  # 设置 x 轴标签字体大小
    plt.ylabel('Category', fontsize=22)  # 设置 y 轴标签字体大小
    plt.title(title, fontsize=24)  # 也可以设置标题的字体大小
    plt.xticks(fontsize=16)  # 设置 x 轴刻度标签的字体大小
    plt.yticks(fontsize=16)  # 设置 y 轴刻度标签的字体大小
    plt.gca().invert_yaxis()  # 将最高的值放在顶部
    plt.tight_layout()

    # 保存图片
    plt.savefig('data/' + filename)
    plt.close()

# 读取 CSV 文件并进行处理
file_path = 'data/output.csv'  # 替换为你的 CSV 文件路径
df = read_csv_file(file_path)

# 统计并绘制条形图
site_sorted, province_sorted, city_sorted, district_sorted = count_values(df)

# 保存前10个统计结果为条形图
plot_and_save_bar_chart(site_sorted, 'Top 10 Sites by Count', 'top_10_sites.png')
plot_and_save_bar_chart(province_sorted, 'Top 10 Provinces by Count', 'top_10_provinces.png')
plot_and_save_bar_chart(city_sorted, 'Top 10 Cities by Count', 'top_10_cities.png')
plot_and_save_bar_chart(district_sorted, 'Top 10 Districts by Count', 'top_10_districts.png')

print("Charts saved as 'top_10_sites.png', 'top_10_provinces.png', 'top_10_cities.png', 'top_10_districts.png'")
