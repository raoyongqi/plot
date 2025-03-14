import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']

def read_js_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        js_content = file.read()

    json_content = re.sub(r'export const capitals = ', '', js_content).strip()
    json_content = json_content.rstrip(';')

    return json.loads(json_content)


def count_values(capitals_data):
    province_counter = Counter()
    city_counter = Counter()
    district_counter = Counter()

    for item in capitals_data:
        province = item.get('province')
        city = item.get('city')
        district = item.get('district')

        if province:
            province_counter[province] += 1
        if city:
            city_counter[city] += 1
        if district:
            district_counter[district] += 1

    province_sorted = province_counter.most_common()
    city_sorted = city_counter.most_common()
    district_sorted = district_counter.most_common()

    return province_sorted, city_sorted, district_sorted

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

file_path = 'pie-app/src/data/site2.js'
capitals_data = read_js_file(file_path)

province_sorted, city_sorted, district_sorted = count_values(capitals_data)

plot_and_save_bar_chart(province_sorted, 'Top 10 Provinces by Count', 'top_10_provinces.png')
plot_and_save_bar_chart(city_sorted, 'Top 10 Cities by Count', 'top_10_cities.png')
plot_and_save_bar_chart(district_sorted, 'Top 10 Districts by Count', 'top_10_districts.png')

print("Charts saved as 'top_10_provinces.png', 'top_10_cities.png', and 'top_10_districts.png'")
