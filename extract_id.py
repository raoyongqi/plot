import pandas as pd

file_path = 'rank1.xlsx'
df = pd.read_excel(file_path)

song_list = df[['歌单名称', '歌单地址']]

song_list['歌单地址ID'] = song_list['歌单地址'].str.extract(r'id=(\d+)')

id_list = song_list['歌单地址ID'].dropna().astype(str).tolist()

with open('song_ids.txt', 'w') as f:
    for song_id in id_list:
        f.write(song_id + '\n')

print("歌单地址ID已保存到 song_ids.txt 文件中")
