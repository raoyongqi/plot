import psutil

# 获取虚拟内存信息
memory_info = psutil.virtual_memory()

# 获取交换空间信息
swap_info = psutil.swap_memory()

# 输出系统内存信息
print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")  # 总内存
print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")    # 已用内存
print(f"Free Memory: {memory_info.free / (1024 ** 3):.2f} GB")    # 可用内存
print(f"Memory Usage: {memory_info.percent}%")                     # 内存使用百分比

# 输出交换空间信息
print(f"Total Swap: {swap_info.total / (1024 ** 3):.2f} GB")      # 交换空间总量
print(f"Used Swap: {swap_info.used / (1024 ** 3):.2f} GB")        # 已用交换空间
print(f"Free Swap: {swap_info.free / (1024 ** 3):.2f} GB")        # 空闲交换空间
print(f"Swap Usage: {swap_info.percent}%")                         # 交换空间使用百分比
