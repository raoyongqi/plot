# 加载必要的包1

if (!requireNamespace("geodata", quietly = TRUE)) {
  # If not installed, install it
  install.packages("geodata")
}

library(geodata)
library(openxlsx)
library(raster)  # 添加对 raster 包的引用
library(utils)   # 添加对 utils 包的引用

sys_info <- Sys.info()

if (!is.null(sys_info)) {
  if (sys_info['sysname'] == 'Windows') {
    print("系统是Windows")
    list.files("C:/Users/r/Desktop/work12/data")
    
    if (!dir.exists(normalizePath("C:/Users/r/Desktop/work12/data"))) {
      stop("目录不存在")
    }
    setwd(normalizePath("C:/Users/r/Desktop/work12/data"))
    
  } else if (sys_info['sysname'] == 'Linux') {
    os_release <- readLines('/etc/os-release')
    if (any(grepl("Ubuntu", os_release))) {
      print("系统是Ubuntu")
      setwd("~/Desktop/getData/work12/data")

      
    } else {
      print("系统是其他Linux发行版")
    }
  } else {
    print("系统不是Windows或Ubuntu")
  }
} else {
  print("无法获取系统信息")
}

library(readxl)

# 设置工作目录

# 读取 Excel 文件中的经纬度数据
data <- read_excel("merge.xlsx")
names(data) <- tolower(names(data))

# 检查是否包含 'lon' 和 'lat' 列
if(!all(c('lon', 'lat') %in% names(data))) {
  stop("Excel 文件需要包含 'lon' 和 'lat' 列")
}

# 定义要下载的气候变量
variables <- c("bio", "elev", "prec", "srad", "tavg", "tmax", "tmin", "vapr", "wind")

# 定义一个函数来下载和提取单个样点的数据
download_and_extract <- function(lon, lat, var, res = 5) {
  clim_data <- geodata::worldclim_global(var = var, res = res, path = ".")
  return(raster::extract(clim_data, matrix(c(lon, lat), ncol = 2)))
}

# 初始化结果数据框
result <- data

# 迭代每个气候变量，下载并提取数据
for (var in variables) {
  cat("Processing variable:", var, "\n")
  
  # 初始化进度条
  pb <- txtProgressBar(min = 0, max = nrow(data), style = 3)
  
  all_clim_values <- NULL
  
  for (i in 1:nrow(data)) {
    lon <- data$lon[i]
    lat <- data$lat[i]
    
    clim_values <- download_and_extract(lon, lat, var)
    
    if (is.null(all_clim_values)) {
      all_clim_values <- clim_values
    } else {
      all_clim_values <- rbind(all_clim_values, clim_values)
    }
    
    # 更新进度条
    setTxtProgressBar(pb, i)
  }
  
  # 关闭进度条
  close(pb)
  
  # 将提取到的数据与原始样点数据合并
  result <- cbind(result, all_clim_values)
}

# 保存结果到新的 Excel 文件
output_file_path <- "climate_data.xlsx"
write.xlsx(result, output_file_path, rowNames = FALSE)
# 保存为 CSV 文件，使用 UTF-8 编码
csv_output_file_path <- "data/climate_data.csv"
write.csv(result, file = csv_output_file_path, row.names = FALSE, fileEncoding = "UTF-8")

# 完成
cat("气候数据已成功保存到", output_file_path, "和", csv_output_file_path, "\n")
