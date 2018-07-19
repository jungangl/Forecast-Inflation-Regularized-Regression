library(forecast)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(multDM)

level = 4
h = 3
J = 301

df_in <- read_csv(paste0("../../data/result-forc-indi/level", level,
                         "-h", h, "-J", J, 
                         "/combined", 
                         ".csv"))
y_real <- df_in$REAL
df_in <- df_in %>% select(-REAL)

df_out <- data.frame() 
df_out$model <- colnames(df_in)


DM.test(df$LAS, df$RDG, df$REAL, loss.type = "SE", h, c = FALSE, H1 = "same")

ncol(df) - 1
colnames(df_in)
