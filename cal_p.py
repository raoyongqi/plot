import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rcParams

# 设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 显示中文

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def count_values(df):
    site_counter = Counter()
    province_counter = Counter()
    city_counter = Counter()
    district_counter = Counter()

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

    site_sorted = site_counter.most_common()
    province_sorted = province_counter.most_common()
    city_sorted = city_counter.most_common()
    district_sorted = district_counter.most_common()

    return site_sorted, province_sorted, city_sorted, district_sorted

def plot_and_save_bar_chart(data, title, filename):

    top_10 = data[:10]
    labels, values = zip(*top_10)

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    
    plt.xlabel('Count', fontsize=22)
    plt.ylabel('Category', fontsize=22)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig('data/' + filename)
    plt.close()


def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['Category', 'Count'])
    df.to_csv('data/' + filename, index=False)

file_path = 'data/output.csv'
df = read_csv_file(file_path)

site_sorted, province_sorted, city_sorted, district_sorted = count_values(df)

plot_and_save_bar_chart(site_sorted, 'Top 10 Sites by Count', 'top_10_sites.png')
plot_and_save_bar_chart(province_sorted, 'Top 10 Provinces by Count', 'top_10_provinces.png')
plot_and_save_bar_chart(city_sorted, 'Top 10 Cities by Count', 'top_10_cities.png')
plot_and_save_bar_chart(district_sorted, 'Top 10 Districts by Count', 'top_10_districts.png')

save_to_csv(site_sorted, 'site_counts.csv')
save_to_csv(province_sorted, 'province_counts.csv')
save_to_csv(city_sorted, 'city_counts.csv')
save_to_csv(district_sorted, 'district_counts.csv')

print("Charts saved as 'top_10_sites.png', 'top_10_provinces.png', 'top_10_cities.png', 'top_10_districts.png'")
print("Data saved as 'site_counts.csv', 'province_counts.csv', 'city_counts.csv', 'district_counts.csv'")
