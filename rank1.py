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

# 假设歌单地址在 '歌单地址' 列
for url in df['歌单地址']:  # 替换为实际的列名
    match = re.search(pattern, str(url))
    if match:
        playlist_id = match.group(1)
        print(f"下载歌单 ID: {playlist_id}")

        # 执行命令：pnpx music-list-downloader@latest download-list +id +downloadDir
        try:
            script_path = r"C:\Users\r\AppData\Roaming\npm\pnpx.ps1"

            # 执行 PowerShell 脚本
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path, "music-list-downloader@latest", "download-list", playlist_id],
                check=True,
                cwd=target_directory  # 设置工作目录
            )
            print(f"歌单 ID {playlist_id} 下载完成！")
        except subprocess.CalledProcessError as e:
            print(f"下载歌单 ID {playlist_id} 时发生错误: {e}")
