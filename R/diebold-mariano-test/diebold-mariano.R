library(forecast)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(multDM)


level = 4
for (h in c(3, 6, 12)){
  for (J in c(301, 373, 493)){
    df_in <- read_csv(paste0("../../data/result-forc-indi/level", level,
                             "-h", h, "-J", J, 
                             "/combined", 
                             ".csv"))
    y_real <- df_in$REAL
    df_in <- df_in %>% select(-REAL)
    p_value <- colnames(df_in)
    df_out <- data.frame(p_value)
    models <- p_value
    num_model <- length(models)
    
    for (model_col in models){
      dm_vec <- numeric(num_model)
      for (r in 1:num_model){
        model_row <- models[r]
        dmtest <- DM.test(df_in[model_col], df_in[model_row], 
                          y_real, loss.type = "SE", 
                          h, c = TRUE, H1 = "same")
        dm_vec[r] <- dmtest$p.value
      }
      df_out$newcol <- dm_vec
      colnames(df_out)[colnames(df_out)=="newcol"] <- model_col
    }
    write.csv(file = paste0("../../data/diebold-mariano/level4-h", h, "-J", J, ".csv"), df_out, row.names=FALSE)
  }
}



