import psutil

memory_info = psutil.virtual_memory()

swap_info = psutil.swap_memory()

print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
print(f"Free Memory: {memory_info.free / (1024 ** 3):.2f} GB")
print(f"Memory Usage: {memory_info.percent}%")

print(f"Total Swap: {swap_info.total / (1024 ** 3):.2f} GB")
print(f"Used Swap: {swap_info.used / (1024 ** 3):.2f} GB")
print(f"Free Swap: {swap_info.free / (1024 ** 3):.2f} GB")
print(f"Swap Usage: {swap_info.percent}%")
