# DYNATE
DYNATE package


## Installation guidance
In R, run the following block of codes:

```{r, include = FALSE}
remotes::install_github(
  repo = "jichunxie/DYNATE",
  build_vignettes = TRUE,
  build_manual = TRUE)
```


A tutorial of this package can be found in this repository under "DYNATE/vignettes/" or running the following block of codes in R:

```{r, include = FALSE}
library(DYNATE)
browseVignettes("DYNATE")
```

## Important Update: DYNATE.R is now renamed to DYNATE1.1.R 

We have renamed the updated version of DYNATE.R to DYNATE1.1.R, which can be located in the "R" folder of this repository. For a detailed tutorial on how to use DYNATE1.1, please refer to the "DYNATE/vignettes/" directory within this repository.

## Changes in DYNATE1.1 compared to Version 1: 

## Input Data Changes: 
The input data format has been modified. It now consists of two main components: a Subject-SNP table and a SNP meta table.
## Output Data Changes: 
In Version 1.1, there is a significant alteration in the output data structure. Instead of only providing the SNP ID detected in the previous version, we now output p-values for each layer of analysis.
The output now includes p-values for each layer detected, providing more comprehensive information. You can find specific details and examples in the tutorial.
These updates aim to enhance the usability and interpretability of DYNATE1.1. For a comprehensive understanding of the changes and usage instructions, please refer to the tutorial available in the "DYNATE/vignettes/" directory.
