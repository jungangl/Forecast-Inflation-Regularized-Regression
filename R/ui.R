library(shiny)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(reshape2)
# 
# read_csv("../data/with-real/actual-forc-data.csv") %>% 
#   mutate(time = row_number()) %>% 
#   select(c("REAL", "ARM"))

ui <- fluidPage(
  sidebarLayout(
    
    sidebarPanel(
      checkboxGroupInput(
      "models", "Time Series Comparison:",
      c(
      "AR1 Model" = "ARM",
      "OLS Model" = "OLS",
      "Model Averaging" = "MAG",
      "Dynamic Factor Model" = "DFM",
      "Ridge Model" = "RDG",
      "Ridge Model 2" = "RDG2",
      "LASSO Model" = "LAS",
      "LASSO Model 2" = "LAS2",
      "Random Forest Regression" = "RFM",
      "Bayesian Model Averaging" = "BMA",
      "Ensemble" = "ESMB"))
    ),
  
  mainPanel(
    plotOutput("timeplot")
    )
  )
)

  
server <- function(input, output) {
  tsdata <- reactive({
    print(input$models)
    check <- read_csv("../data/with-real/actual-forc-data.csv") %>% 
      select(REAL, input$models) %>% 
      mutate(time = row_number()) %>% 
      melt(id = "time")
    })
  
  output$timeplot <- renderPlot({
    ggplot(tsdata(), aes(x = time, y = value, color = variable)) + 
      #geom_line(aes(y = REAL, color = "REAL"), size = 1) +
      geom_line() +
      theme_minimal() + 
      theme(legend.title=element_blank()) + 
      labs(y = "Inflation", x = "Time")
    })
  }

shinyApp(ui, server)
