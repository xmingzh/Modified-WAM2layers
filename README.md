# Modified-WAM2layers
 
The origianl Python code for the WAM-2layers model was retrieved from GitHub (https://github.com/ruudvdent/WAM2layersPython). It was further modified to fit the ERA5 data.
In the original WAM-2layers, the time step of the calculation is reduced to 15 minutes to reduce the Courant number given the ERA-Interim data. The spatial resolutions of ERA5 are higher than that of ERA-Interim. Hence, a smaller time step is needed for the calculation and a 10 minutes time step was used in this modification. Notedly, smaller time step may be better, but it takes more time for the calculation.
