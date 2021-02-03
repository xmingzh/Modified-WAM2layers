# -*- coding: utf-8 -*-

#%% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as cm
from netCDF4 import Dataset

#%%
path = r'D:\Github\WAM2layers_for_ERA5\landseamask.nc'
landmask = np.squeeze(Dataset(path, mode='r')['lsm'][:,:,:])
lat = Dataset(path, mode='r')['latitude'][:]
lon = Dataset(path, mode='r')['longitude'][:]

# land sea mask
lsm=np.loadtxt('Landseamask.csv',delimiter=',') 

# Figure
plt.figure() 
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

ax.coastlines(linewidth=0.4)

x_shift,y_shift = np.meshgrid(lon,lat)
x = x_shift - 0.125
y = y_shift + 0.125

lol = plt.pcolormesh(x,y,lsm,transform=ccrs.PlateCarree(), cmap=cm.cm.coolwarm, vmin=0,vmax=1)

plt.savefig('lsm_ERA5.PNG', format='PNG', dpi=200)
