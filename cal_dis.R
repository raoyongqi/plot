# PL <- Community_data2 %>% mutate(
#   Severity = (Disease_0*0 + Disease_1*1 +Disease_2*2+Disease_3*3+ Disease_4*4+
#                 Disease_5*5)/
#     6/(Disease_0+Disease_1+Disease_2+Disease_3+Disease_4+Disease_5)) %>% 
#   groupby(Treatment,Site,plot) %>% summarise(PL= sum(Severity*species_biomass,na.rm=True)/mean(Plot_Biomass))

# 分母是0？？

# 加载 openxlsx 和 dplyr 包
library(openxlsx)
library(dplyr)
setwd("C:/Users/r/Desktop/work12")

# 读取 Excel 文件
df <- read.xlsx("data/ruqinfeiruqin.xlsx", sheet = 1)
# 查看数据框的列名
colnames(df)
# 

df <- df %>%
  rename(
    Disease_0 = D0,
    Disease_1 = D1,
    Disease_2 = D2,
    Disease_3 = D3,
    Disease_4 = D4,
    Disease_5 = D5
  )
df[is.na(df)] <- 0


# 计算 Severity
df <- df %>%
  mutate(Severity = (Disease_0*0 + Disease_1*1 + Disease_2*2 + Disease_3*3 + Disease_4*4 + Disease_5*5) /
           (6 * (Disease_0 + Disease_1 + Disease_2 + Disease_3 + Disease_4 + Disease_5)))


# 查看结果
print(df)

