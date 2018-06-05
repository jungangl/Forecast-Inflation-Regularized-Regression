# Forecasting inflation with disaggregated CPI's using machine learning techniques

This repo contains data and *Julia* code for an illustration of forecasting inflation with disaggregated price data. Click on the "fork" button at the very top right of the page to create an independent copy of the repo within your own GitHub account. Alternately, click on the green "clone or download" button just below that to download the repo to your local computer.

The main file for conducting the analysis is `julia/real-data-forecast.jl` and `julia/generated-data-forecast.jl`. Here `julia/real-data-forecast.jl` uses real data, and `julia/generated-data-forecast.jl` uses generated data (from a dynamic factor model). Each file contains self-explanatory code for easily reproducing the results from the eight different models:
* Auto-regressive model
* OLS model
* Model averaging with equal weights
* Dynamic factor model
* Ridge model
* LASSO model
* Random forrest model
* Bayesian model averaging

## Requirements

The entire analysis is conducted in the *Julia* programming environment. *Julia* is free, open-source and available for download [here](https://julialang.org/downloads/). We highly recommend running *Julia* in the Atom IDE, which you can also download for free [here](https://atom.io/).

You will need to install a number of external *Julia* packages to run the code successfully. These are listed at the top of the main `julia/generated-data-forecast.jl` file. An easy way to ensure that you have the correct versions of all the packages is to run the following code chunk in your *Julia* console:

```
Pkg.add("CSV")
Pkg.add("GLMNet")
Pkg.add("Combinatorics")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("ScikitLearn")
Pkg.update()
Pkg.build()
```


## Performance

The core analysis in this paper involves a series of computationally intensive parameter-tuning procesess. The code is optimized to run in parallel and will automatically exploit any multi-core capability on your machine. We recommend use a remote server (super computers or google cloud) with a large number of cores and high-performance CPUs. (Grant McDermott has a very nice tutorial on how to set up a virtual machine, click [here](http://grantmcdermott.com/2017/05/30/rstudio-server-compute-engine/) to learn from the tutorial.

## Problems

If you have any trouble running the code, or find any errors, please file an issue on this repo and we'll look into it.

## License
The software code so far is still in a private repository. Once it is finished we will publish it under the MIT license.
