
#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from getconstants import getconstants
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import matplotlib as cm
import os
from timeit import default_timer as timer
import calendar
import datetime
#%% BEGIN OF INPUT (FILL THIS IN)
years = np.arange(1980,2019) #fill in the years
yearpart = np.arange(0,366) # for a full (leap)year fill in (365,-1,-1)
# Manage the extent of your dataset (FILL THIS IN)
# Define the latitude and longitude cell numbers to consider and corresponding lakes that should be considered part of the land
latnrs = np.arange(196,433) # 43N - -10N
lonnrs = np.arange(880,1400) #40E - 180E

# obtain the constants
invariant_data = r'/public/home/mzxiao/ERA5/landseamask.nc'#invariants
latitude,longitude,lsm,g,density_water,A_gridcell,L_N_gridcell,L_S_gridcell,L_EW_gridcell = getconstants(latnrs,lonnrs,invariant_data)
A_gridcell2D = np.tile(A_gridcell,len(longitude))

# BEGIN OF INPUT 2 (FILL THIS IN)
timetracking = 0 # 0 for not tracking time and 1 for tracking time
# Region = lsm[latnrs,:][:,lonnrs]
# The focus Region
Region_s=np.zeros(np.shape(lsm))
# The selected region
Region_s[265:274,1169:1180]=1 # N:21.75-23.75 E: 112.25-114.75
lsm_s=lsm*Region_s
Region=lsm_s[latnrs,:][:,lonnrs]
Region[Region>0.8]=1 # Change the lake also as land, which is not in lsm

interdata_folder = r'/public/home/mzxiao/WAM2layersPython_modify/interdata'
sub_interdata_folder = os.path.join(interdata_folder, 'continental_backward')
output_folder = os.path.join(interdata_folder, 'output')
# input_folder = r'/public/home/mzxiao/WAM2layersPython_modify/interdata/continental_backward'

# END OF INPUT

#%% Datapaths (FILL THIS IN)

def data_path(yearnumber,a,years,timetracking):
    load_Sa_track = os.path.join(sub_interdata_folder, str(yearnumber) + '-' + str(a) + 'Sa_output_track.mat')
    load_Sa_time = os.path.join(sub_interdata_folder, str(yearnumber) + '-' + str(a) + 'Sa_output_time.mat')
    save_path = os.path.join(output_folder, 'E_track_continental_full' + str(years[0]) + '-' + str(years[-1]) + '-timetracking' + str(timetracking) + '.mat')
    return load_Sa_track,load_Sa_time,save_path

# Define variables
E_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
E_track_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
P_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
north_loss_year_month = np.zeros((len(years),12,1,len(longitude)))
south_loss_year_month = np.zeros((len(years),12,1,len(longitude)))
west_loss_year_month = np.zeros((len(years),12,len(latitude)))
east_loss_year_month = np.zeros((len(years),12,len(latitude)))
water_lost_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
Fa_E_down_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
Fa_E_top_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
Fa_N_down_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
Fa_N_top_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))
if timetracking == 1:
    E_time_year_month = np.zeros((len(years),12,len(latitude),len(longitude)))

#%%  Calculations 
startyear = years[0]
for y in years:
    ly = int(calendar.isleap(y))
    final_time = 364+ly  
    num_day=365+ly
    if y==2018:
         final_time=363 
    # Define variables
    E_year = np.zeros((num_day,len(latitude),len(longitude)))
    E_track_year = np.zeros((num_day,len(latitude),len(longitude)))
    P_year = np.zeros((num_day,len(latitude),len(longitude))) 
    north_loss_year = np.zeros((num_day,1,len(longitude)))
    south_loss_year = np.zeros((num_day,1,len(longitude)))
    west_loss_year = np.zeros((num_day,len(latitude),1))
    east_loss_year = np.zeros((num_day,len(latitude),1))
    water_lost_year = np.zeros((num_day,len(latitude),len(longitude)))
    Fa_E_down_year = np.zeros((num_day,len(latitude),len(longitude)))
    Fa_E_top_year = np.zeros((num_day,len(latitude),len(longitude)))
    Fa_N_down_year = np.zeros((num_day,len(latitude),len(longitude)))
    Fa_N_top_year = np.zeros((num_day,len(latitude),len(longitude)))    
    
    for a in yearpart:
        datapath = data_path(y,a,years,timetracking)
        if a>final_time:
            pass
        else:
            # load output data
            loading_st=sio.loadmat(datapath[0],verify_compressed_data_integrity=False)
            E_day=loading_st['E_day']
            E_track_day=loading_st['E_track_day']
            P_day=loading_st['P_day']
            north_loss_day=loading_st['north_loss_day']
            south_loss_day=loading_st['south_loss_day']
            west_loss_day=loading_st['west_loss_day']
            east_loss_day=loading_st['east_loss_day']
            water_lost_day=loading_st['water_lost_day']
            Fa_E_down_day=loading_st['Fa_E_down_day']
            Fa_E_top_day=loading_st['Fa_E_top_day']
            Fa_N_down_day=loading_st['Fa_N_down_day']
            Fa_N_top_day=loading_st['Fa_N_top_day']

            E_year[a,:,:] = E_day
            E_track_year[a,:,:] = E_track_day
            P_year[a,:,:] = P_day
            north_loss_year[a,:,:] = north_loss_day
            south_loss_year[a,:,:] = south_loss_day
            west_loss_year[a,:,:] = west_loss_day
            east_loss_year[a,:,:] = east_loss_day
            water_lost_year[a,:,:] = water_lost_day
            Fa_E_down_year[a,:,:] = Fa_E_down_day
            Fa_E_top_year[a,:,:] = Fa_E_top_day
            Fa_N_down_year[a,:,:] = Fa_N_down_day
            Fa_N_top_year[a,:,:] = Fa_N_top_day

            if timetracking==1:
                loading_t=sio.loadmat(datapath[1],verify_compressed_data_integrity=False)
                E_time_year = np.zeros((num_day,len(latitude),len(longitude)))
                E_time_day=loading_t['E_time_day']
                E_time_year[a,:,:] = E_time_day


    # values per month        
    for m in range(12):
        first_day = int(datetime.date(y,m+1,1).strftime("%j"))-1
        last_day = int(datetime.date(y,m+1,calendar.monthrange(y,m+1)[1]).strftime("%j"))-1
        # days = np.arange(first_day,last_day+1)-1 # -1 because Python is zero-based
        days=np.squeeze(np.where(np.logical_and(yearpart>=first_day, yearpart<=last_day)))

        E_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(E_year[days,:,:], axis = 0)))
        E_track_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(E_track_year[days,:,:], axis = 0)))
        P_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(P_year[days,:,:], axis = 0)))
        
        north_loss_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(north_loss_year[days,:,:], axis = 0)))
        south_loss_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(south_loss_year[days,:,:], axis = 0)))
        west_loss_year_month[y-startyear,m,:] = (np.squeeze(np.sum(west_loss_year[days,:,:], axis = 0)))
        east_loss_year_month[y-startyear,m,:] = (np.squeeze(np.sum(east_loss_year[days,:,:], axis = 0)))
        water_lost_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(water_lost_year[days,:,:], axis = 0)))

        Fa_E_down_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(Fa_E_down_year[days,:,:], axis = 0)))
        Fa_E_top_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(Fa_E_top_year[days,:,:], axis = 0)))
        Fa_N_down_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(Fa_N_down_year[days,:,:], axis = 0)))
        Fa_N_top_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(Fa_N_top_year[days,:,:], axis = 0)))

        if timetracking==1:
            with np.errstate(divide='ignore', invalid='ignore'):
                E_time_year_month[y-startyear,m,:,:] = (np.squeeze(np.sum(E_time_year[days,:,:] * E_track_year[days,:,:], axis=0))
                 / np.squeeze(E_track_year_month[y-startyear,m,:,:]))                
            # remove nans                
            where_are_NaNs = np.isnan(E_time_year_month)
            E_time_year_month[where_are_NaNs] = 0
        elif timetracking==0:
            E_time_year_month = 0

# Save data
sio.savemat(datapath[2],
           {'E_year_month':E_year_month,'E_track_year_month':E_track_year_month,'P_year_month':P_year_month,
            'north_loss_year_month':north_loss_year_month,'south_loss_year_month':south_loss_year_month,
            'east_loss_year_month':east_loss_year_month,'west_loss_year_month':west_loss_year_month,
            'water_lost_year_month':water_lost_year_month,'Fa_E_down_year_month':Fa_E_down_year_month,
            'Fa_E_top_year_month':Fa_E_top_year_month,'Fa_N_down_year_month':Fa_N_down_year_month,
             'Fa_N_top_year_month':Fa_N_top_year_month,'E_time_year_month':E_time_year_month},do_compression=True)