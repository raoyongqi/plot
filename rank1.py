import pandas as pd
import re
import subprocess
import os

df = pd.read_excel('rank1.xlsx')

pattern = r"id=(\d+)"

target_directory = r'C:\Users\r\Downloads\nmd'

for url in df['歌单地址']:


    match = re.search(pattern, str(url))
    if match:
        playlist_id = match.group(1)
        print(f"下载歌单 ID: {playlist_id}")

        try:
            script_path = r"C:\Users\r\AppData\Roaming\npm\pnpx.ps1"

            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path, "music-list-downloader@latest", "download-list", playlist_id],
                check=True,
                cwd=target_directory
            )
            
            print(f"歌单 ID {playlist_id} 下载完成！")
        except subprocess.CalledProcessError as e:
            print(f"下载歌单 ID {playlist_id} 时发生错误: {e}")
