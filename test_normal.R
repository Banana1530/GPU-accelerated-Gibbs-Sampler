library(readr)
suppressMessages(beta_result <- read_table2("result_beta.txt", 
                           col_names = FALSE))
X = beta_result
options(scipen=999)
X = X[,c(1,2,3)]
print(cov(X),digits = 4,scientific = FALSE)
print(colMeans(X),digits = 4,scientific = FALSE)
