import pandas as pd

# 读取Excel文件
file_path = 'rank1.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 提取歌单名称和歌单地址
song_list = df[['歌单名称', '歌单地址']]

# 提取歌单地址中的ID部分
song_list['歌单地址ID'] = song_list['歌单地址'].str.extract(r'id=(\d+)')

# 只保留歌单地址ID这一列，并转换为列表
id_list = song_list['歌单地址ID'].dropna().astype(str).tolist()

# 将ID列表写入txt文件
with open('song_ids.txt', 'w') as f:
    for song_id in id_list:
        f.write(song_id + '\n')

print("歌单地址ID已保存到 song_ids.txt 文件中")
