import pandas as pd
import re
import subprocess
import os

# 读取 Excel 文件
df = pd.read_excel('rank1.xlsx')

# 正则表达式提取 id
pattern = r"id=(\d+)"

# 设置目标工作目录
target_directory = r'C:\Users\r\Downloads\nmd'

# 目标歌曲 ID
target_song_id = 138911210
found_song = None  # 用于存储找到的歌曲名

# 查找目标歌曲在 DataFrame 中的位置
target_song_row = df[df['歌单地址'].str.contains(str(target_song_id), na=False)]

if not target_song_row.empty:
    # 获取目标歌曲所在的行号
    row_index = target_song_row.index[0]
    print(f"目标歌曲 ID {target_song_id} 位于第 {row_index + 1} 行（Excel 中的第 {row_index + 1} 行，索引从 0 开始）。")
else:
    print(f"未找到目标歌曲 ID {target_song_id} 在 Excel 中的任何位置。")