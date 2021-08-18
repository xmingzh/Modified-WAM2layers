# Modified-WAM2layers
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4796962.svg)](http://doi.org/10.5281/zenodo.4796962)
 
The origianl Python code for the WAM-2layers model was retrieved from GitHub (https://github.com/ruudvdent/WAM2layersPython). It was further modified to fit the ERA5 data.

In the original WAM-2layers, the time step of the calculation is reduced to 15 minutes to reduce the Courant number given the ERA-Interim data. The spatial resolutions of ERA5 are higher than that of ERA-Interim. Hence, a smaller time step is needed for the calculation and a 10 minutes time step was used in this modification. Notedly, smaller time step may be better, but it takes more time for the calculation.

In the original WAM-2layers, the model level data was used. The official document of ERA5 data (https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5) stated that the model level data are not stored on spinning disk, it is very slow to download this data. Thereby, the pressure level data was used for the Modified-WAM2layers.For the pressure level data, the division of top and bottom layers is a little more complicated in the place with low surface pressure. Besides, the pressure leve data below the surface is meaningless. Hence, the surface pressure was further used to mask out the extrapolated pressure level data.
# References
Xiao, M., & Cui, Y. (2021). Source of evaporation for the seasonal precipitation in the Pearl River Delta, China. Water Resources Research, 57, e2020WR028564. https://doi.org/10.1029/2020WR028564
