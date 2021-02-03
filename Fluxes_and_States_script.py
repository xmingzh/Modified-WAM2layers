#!/usr/bin/env python

#%% Import libraries
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
import calendar
from getconstants import getconstants
from timeit import default_timer as timer
import os
import datetime

#%% sub functions
def getW(latnrs,lonnrs,final_time,a,time_low,time_up,
    density_water,latitude,longitude,g,A_gridcell):
    
    if a != final_time: # not the end of the year
        # To select the data index by match their date
        # and this can help to avoid the error of date not match in different datasets
        time=Dataset(datapath[2], mode = 'r').variables['time'][:]
        step=time[2]-time[1]
        time_up=time_up+step
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        # time_index=np.arange(begin_time,(begin_time+count_time+1))
        
        # specific humidity (state at 00.00, 06.00, 12.00, 18.00)
        q = Dataset(datapath[0], mode = 'r').variables['q'][time_index,:,latnrs,lonnrs] #kg/kg
        
        # total column water 
        tcw_ERA = Dataset(datapath[2], mode = 'r').variables['tcw'][time_index,latnrs,lonnrs] #kg/m2
    
    else: #end of the year
        time=Dataset(datapath[2], mode = 'r').variables['time'][:]
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        index_insert=len(time_index)
        # time_index=np.arange(begin_time,(begin_time+count_time))
        # surface pressure (state at 00.00, 06.00, 12.00, 18.00)
        # sp_first = Dataset(datapath[0], mode = 'r').variables['sp'][begin_time:(begin_time+count_time),latnrs,lonnrs]
        # sp = np.insert(sp_first,[4],(Dataset(datapath[1], mode = 'r').variables['sp'][0,latnrs,lonnrs]), axis = 0) #Pa
        
        # specific humidity (state at 00.00 06.00 12.00 18.00)
        q_first = Dataset(datapath[0], mode = 'r').variables['q'][time_index,:,latnrs,lonnrs]
        q = np.insert(q_first,[index_insert],(Dataset(datapath[1], mode = 'r').variables['q'][0,:,latnrs,lonnrs]), axis = 0) #kg/kg
        
        # total column water 
        tcw_ERA_first = Dataset(datapath[2], mode = 'r').variables['tcw'][time_index,latnrs,lonnrs]
        tcw_ERA = np.insert(tcw_ERA_first,[index_insert],Dataset(datapath[3], mode = 'r').variables['tcw'][0,latnrs,lonnrs],axis = 0)

    # make cwv vector
    # The pressure level for the download data
    pressure_level= Dataset(datapath[0], mode = 'r').variables['level'][:]
    # To select the closest pressure level to 812.83 hPa to divide the column air into the bottom and top layers
    diff=pressure_level-812.83
    boundary=np.argmin(np.abs(diff))-1

    q_swap = np.swapaxes(q,0,1)
    cwv_swap = np.zeros((len(pressure_level)-1, len(q), len(latitude), len(longitude))) #kg/m2
    for n in range(len(pressure_level)-1):
        cwv_swap[n] = (0.5*np.squeeze(q_swap[n]+q_swap[n+1])*(pressure_level[n+1]-pressure_level[n])*100) / g # column water vapor = specific humidity * pressure levels length / g [kg/m2]
    cwv = np.swapaxes(cwv_swap,0,1)
    
    # make tcwv vector
    tcwv = np.squeeze(np.sum(cwv,1)) #total column water vapor, cwv is summed over the vertical [kg/m2]
    
    # make cw vector
    cw_swap = np.zeros((np.shape(cwv_swap)))
    for n in range(len(pressure_level)-1):
        cw_swap[n] = (tcw_ERA / tcwv) * np.squeeze(cwv_swap[n])
    cw = np.swapaxes(cw_swap,0,1)
        
    # just a test, return this variable when running tests
    # tcw = np.squeeze(np.sum(cw,1)) #total column water, cw is summed over the vertical [kg/m2]    
    # testvar = tcw_ERA - tcw # should be around zero for most cells
    
    # put A_gridcell on a 3D grid
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    A_gridcell_1_2D = np.reshape(A_gridcell2D, [1,len(latitude),len(longitude)])
    A_gridcell_plus3D = np.tile(A_gridcell_1_2D,[len(q),1,1])
    
    # water volumes
    vapor_top = np.squeeze(np.sum(cwv[:,0:boundary,:,:],1))
    vapor_down = np.squeeze(np.sum(cwv[:,boundary:,:,:],1))
    vapor = vapor_top + vapor_down
    W_top = tcw_ERA * (vapor_top / vapor) * A_gridcell_plus3D / density_water #m3
    W_down = tcw_ERA * (vapor_down / vapor) * A_gridcell_plus3D / density_water #m3
    
    return cw, W_top, W_down

#%% Code
def getwind(latnrs,lonnrs,final_time,a,time_low,time_up):
    # u stands for wind in zonal direction = west-east
    # v stands for wind in meridional direction = south-north 
    if a != final_time: # not the end of the year
        time=Dataset(datapath[4], mode = 'r').variables['time'][:]
        step=time[2]-time[1]
        time_up=time_up+step
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        # read the u-wind data
        u_f = Dataset(datapath[4], mode = 'r').variables['u'][time_index,:,latnrs,lonnrs] #m/s
        
        # read the v-wind data
        v_f = Dataset(datapath[6], mode = 'r').variables['v'][time_index,:,latnrs,lonnrs] #m/s
        
    else: #end of year
        time=Dataset(datapath[4], mode = 'r').variables['time'][:]
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        index_insert=len(time_index)
        # read the u-wind data
        u_f_first = Dataset(datapath[4], mode = 'r').variables['u'][time_index,:,latnrs,lonnrs]
        u_f = np.insert(u_f_first,[index_insert],(Dataset(datapath[5], mode = 'r').variables['u'][0,:,latnrs,lonnrs]), axis = 0)
        
        # read the v-wind data
        v_f_first = Dataset(datapath[6], mode = 'r').variables['v'][time_index,:,latnrs,lonnrs]
        v_f = np.insert(v_f_first,[index_insert],(Dataset(datapath[7], mode = 'r').variables['v'][0,:,latnrs,lonnrs]), axis = 0)
    
    U = u_f
    V = v_f
    
    return U,V

#%% Code
def getFa(latnrs,lonnrs,cw,U,V,time_low,time_up,a,final_time):
    
    if a != final_time: #not the end of the year
        time=Dataset(datapath[2], mode = 'r').variables['time'][:]
        step=time[2]-time[1]
        time_up=time_up+step
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        #get ERA vertically integrated fluxes
        ewvf = Dataset(datapath[2], mode = 'r').variables['p71.162'][time_index,latnrs,lonnrs]
        nwvf = Dataset(datapath[2], mode = 'r').variables['p72.162'][time_index,latnrs,lonnrs]
        eclwf = Dataset(datapath[2], mode = 'r').variables['p88.162'][time_index,latnrs,lonnrs]
        nclwf = Dataset(datapath[2], mode = 'r').variables['p89.162'][time_index,latnrs,lonnrs]
        ecfwf = Dataset(datapath[2], mode = 'r').variables['p90.162'][time_index,latnrs,lonnrs]
        ncfwf = Dataset(datapath[2], mode = 'r').variables['p91.162'][time_index,latnrs,lonnrs]
        
    else: #end of year
        time=Dataset(datapath[2], mode = 'r').variables['time'][:]
        time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
        index_insert=len(time_index)
        # get the last data from next year
        ewvf_first = Dataset(datapath[2], mode = 'r').variables['p71.162'][time_index,latnrs,lonnrs]
        ewvf = np.insert(ewvf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p71.162'][0,latnrs,lonnrs]), axis = 0)
        nwvf_first = Dataset(datapath[2], mode = 'r').variables['p72.162'][time_index,latnrs,lonnrs]
        nwvf = np.insert(nwvf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p72.162'][0,latnrs,lonnrs]), axis = 0)
        eclwf_first = Dataset(datapath[2], mode = 'r').variables['p88.162'][time_index,latnrs,lonnrs]
        eclwf = np.insert(eclwf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p88.162'][0,latnrs,lonnrs]), axis = 0)
        nclwf_first = Dataset(datapath[2], mode = 'r').variables['p89.162'][time_index,latnrs,lonnrs]
        nclwf = np.insert(nclwf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p89.162'][0,latnrs,lonnrs]), axis = 0)
        ecfwf_first = Dataset(datapath[2], mode = 'r').variables['p90.162'][time_index,latnrs,lonnrs]
        ecfwf = np.insert(ecfwf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p90.162'][0,latnrs,lonnrs]), axis = 0)
        ncfwf_first = Dataset(datapath[2], mode = 'r').variables['p91.162'][time_index,latnrs,lonnrs]
        ncfwf = np.insert(ncfwf_first,[index_insert],(Dataset(datapath[3], mode = 'r').variables['p91.162'][0,latnrs,lonnrs]), axis = 0)
        
    ewf = ewvf + eclwf + ecfwf #kg*m-1*s-1
    nwf = nwvf + nclwf + ncfwf #kg*m-1*s-1
    
    #eastward and northward fluxes
    u_swap = np.swapaxes(U,0,1)
    v_swap = np.swapaxes(V,0,1)
    cw_swap=np.swapaxes(cw,0,1)
    kk= len(cw_swap)
    Fa_E_p_swap = np.zeros(np.shape(cw_swap)) 
    Fa_N_p_swap = np.zeros(np.shape(cw_swap)) 
    for n in range(kk):
        Fa_E_p_swap[n] = (0.5*np.squeeze(u_swap[n]+u_swap[n+1])*np.squeeze(cw_swap[n])) 
        Fa_N_p_swap[n] = (0.5*np.squeeze(v_swap[n]+v_swap[n+1])*np.squeeze(cw_swap[n])) 
    Fa_E_p = np.swapaxes(Fa_E_p_swap,0,1)
    Fa_N_p = np.swapaxes(Fa_N_p_swap,0,1)
    # To estimate the boundary for the top and down, around the location of 812.83 hpa
    pressure_level= Dataset(datapath[0], mode = 'r').variables['level'][:]
    diff=pressure_level-812.83
    boundary=np.argmin(np.abs(diff))
    # uncorrected down and top fluxes
    Fa_E_down_uncorr = np.squeeze(np.sum(Fa_E_p[:,boundary:,:,:],1)) #kg*m-1*s-1
    Fa_N_down_uncorr = np.squeeze(np.sum(Fa_N_p[:,boundary:,:,:],1)) #kg*m-1*s-1
    Fa_E_top_uncorr = np.squeeze(np.sum(Fa_E_p[:,0:boundary,:,:],1)) #kg*m-1*s-1
    Fa_N_top_uncorr = np.squeeze(np.sum(Fa_N_p[:,0:boundary,:,:],1)) #kg*m-1*s-1
    
    # correct down and top fluxes
    corr_E1=ewf/(Fa_E_down_uncorr+Fa_E_top_uncorr)
    corr_N1=nwf/(Fa_N_down_uncorr+Fa_N_top_uncorr)
    size=np.size(corr_E1)
    corr_E1=np.reshape(np.minimum(np.maximum(np.reshape(corr_E1, size),0),2),
                        np.shape(corr_E1)) 
    corr_N1=np.reshape(np.minimum(np.maximum(np.reshape(corr_N1, size),0),2),
                        np.shape(corr_N1))

    Fa_E_down = corr_E1 * Fa_E_down_uncorr #kg*m-1*s-1
    Fa_N_down = corr_N1 * Fa_N_down_uncorr #kg*m-1*s-1
    Fa_E_top = corr_E1 * Fa_E_top_uncorr #kg*m-1*s-1
    Fa_N_top = corr_N1 * Fa_N_top_uncorr #kg*m-1*s-1
    
    # make the fluxes during the timestep
    Fa_E_down = 0.5*(Fa_E_down[0:-1,:,:]+Fa_E_down[1:,:,:])
    Fa_N_down = 0.5*(Fa_N_down[0:-1,:,:]+Fa_N_down[1:,:,:])
    Fa_E_top = 0.5*(Fa_E_top[0:-1,:,:]+Fa_E_top[1:,:,:])
    Fa_N_top = 0.5*(Fa_N_top[0:-1,:,:]+Fa_N_top[1:,:,:])
    
    return Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down


#%% Code
def getEP(latnrs,lonnrs,time_low,time_up,latitude,longitude,A_gridcell):
    
    time=Dataset(datapath[8], mode = 'r').variables['time'][:]
    time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
    #(accumulated after the forecast at 00.00 and 12.00 by steps of 3 hours in time
    evaporation = Dataset(datapath[8], mode = 'r').variables['e'][time_index,latnrs,lonnrs] #m
    precipitation = Dataset(datapath[9], mode = 'r').variables['tp'][time_index,latnrs,lonnrs] #m
        
    #delete and transfer negative values, change sign convention to all positive
    precipitation = np.reshape(np.maximum((np.reshape(precipitation, (np.size(precipitation)))
                  + np.maximum(np.reshape(evaporation, (np.size(evaporation))),0.0)),0.0),
                        np.shape(precipitation)) 
    #Evaporation is the accumulated amount of water, 
    # negative values indicate evaporation and positive values indicate condensation
    evaporation = np.reshape(np.abs(np.minimum(np.reshape(evaporation, (np.size(evaporation))),0.0)),
                     np.shape(evaporation))   
    
    #calculate volumes
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    A_gridcell_1_2D = np.reshape(A_gridcell2D, [1,len(latitude),len(longitude)])
    A_gridcell_max3D = np.tile(A_gridcell_1_2D,[len(time_index),1,1])

    E = evaporation * A_gridcell_max3D
    P = precipitation * A_gridcell_max3D

    return E,P

def getEP_ext(latnrs,lonnrs,time_low,time_up,latitude,longitude,A_gridcell):
    # get the extreme precipitation, defined as: >5 mm/h
    time=Dataset(datapath[8], mode = 'r').variables['time'][:]
    time_index=np.squeeze(np.where(np.logical_and(time>=time_low, time<=time_up)))
    #(accumulated after the forecast at 00.00 and 12.00 by steps of 3 hours in time
    evaporation = Dataset(datapath[8], mode = 'r').variables['e'][time_index,latnrs,lonnrs] #m
    precipitation = Dataset(datapath[9], mode = 'r').variables['tp'][time_index,latnrs,lonnrs] #m
        
    #delete and transfer negative values, change sign convention to all positive
    precipitation0 = np.reshape(np.maximum((np.reshape(precipitation, (np.size(precipitation)))
                  + np.maximum(np.reshape(evaporation, (np.size(evaporation))),0.0)),0.0),
                        np.shape(precipitation)) 
    precipitation1=np.reshape(precipitation0, np.size(precipitation0))
    precipitation1[precipitation1<0.005]=0 # Here the unit of precipitation is m
    precipitation=np.reshape(precipitation1,np.shape(precipitation))

    #Evaporation is the accumulated amount of water, 
    # negative values indicate evaporation and positive values indicate condensation
    evaporation = np.reshape(np.abs(np.minimum(np.reshape(evaporation, (np.size(evaporation))),0.0)),
                     np.shape(evaporation))   
    
    #calculate volumes
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    A_gridcell_1_2D = np.reshape(A_gridcell2D, [1,len(latitude),len(longitude)])
    A_gridcell_max3D = np.tile(A_gridcell_1_2D,[len(time_index),1,1])

    E = evaporation * A_gridcell_max3D
    P = precipitation * A_gridcell_max3D

    return E,P

#%% Code
def getrefined(Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,W_top,W_down,E,P,time_reduce,latitude,longitude):
    
    # Reduce the time scale of the variables to 0.1 hour
    # time_reduce=0.1
    len_time=np.int(24/time_reduce)
    #for 1 hourly information
    time1=Dataset(datapath[8], mode = 'r').variables['time'][:]
    step1=time1[2]-time1[1]
    divt1 = step1/time_reduce    
    
    E_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    P_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    for t in range(0,len_time):
        E_small[t] = (1./divt1) * E[np.int(t/divt1)]
        P_small[t] = (1./divt1) * P[np.int(t/divt1)]
    E = E_small                  
    P = P_small
    
    # for 6 hourly info  
    time2=Dataset(datapath[2], mode = 'r').variables['time'][:]
    step2=time2[2]-time2[1]
    divt2 = step2/time_reduce

    W_top_small = np.nan*np.zeros((len_time+1,len(latitude),len(longitude)))
    W_down_small = np.nan*np.zeros((len_time+1,len(latitude),len(longitude)))
    # 
    for t in range(0,len_time):
        W_top_small[t] = W_top[np.int(t/divt2)]+ (t%divt2)/divt2 * (W_top[np.int(t/divt2)+1]-W_top[np.int(t/divt2)])       
        W_down_small[t] = W_down[np.int(t/divt2)] + (t%divt2)/divt2 * (W_down[np.int(t/divt2)+1] - W_down[np.int(t/divt2)]) 
    W_top_small[-1]=W_top[-1]
    W_down_small[-1]= W_down[-1]
    
    W_top = W_top_small                
    W_down = W_down_small

    Fa_E_down_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    Fa_N_down_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    Fa_E_top_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    Fa_N_top_small = np.nan*np.zeros((len_time,len(latitude),len(longitude)))
    for t in range(0,len_time):
        Fa_E_down_small[t] = Fa_E_down[np.int(t/divt2)]
        Fa_N_down_small[t] = Fa_N_down[np.int(t/divt2)]
        Fa_E_top_small[t] = Fa_E_top[np.int(t/divt2)]
        Fa_N_top_small[t] = Fa_N_top[np.int(t/divt2)]

    Fa_E_down = Fa_E_down_small
    Fa_N_down = Fa_N_down_small
    Fa_E_top = Fa_E_top_small
    Fa_N_top = Fa_N_top_small
    
    return Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,E,P,W_top,W_down


#%% Code
def get_stablefluxes(W_top,W_down,Fa_E_top_1,Fa_E_down_1,Fa_N_top_1,Fa_N_down_1,
                        time_reduce,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude,longitude):
    # time_reduce=0.25
    timestep=time_reduce*3600
    len_time=np.int(24/time_reduce)
    #redefine according to units
    Fa_E_top_kgpmps = Fa_E_top_1
    Fa_E_down_kgpmps = Fa_E_down_1
    Fa_N_top_kgpmps = Fa_N_top_1
    Fa_N_down_kgpmps = Fa_N_down_1
    
    #convert to m3
    Fa_E_top = Fa_E_top_kgpmps * timestep * L_EW_gridcell / density_water # [kg*m^-1*s^-1*s*m*kg^-1*m^3]=[m3]
    Fa_E_down = Fa_E_down_kgpmps * timestep * L_EW_gridcell / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]

    Fa_N_top_swap = np.zeros((len(latitude),len_time,len(longitude)))
    Fa_N_down_swap = np.zeros((len(latitude),len_time,len(longitude)))
    Fa_N_top_kgpmps_swap = np.swapaxes(Fa_N_top_kgpmps,0,1)
    Fa_N_down_kgpmps_swap = np.swapaxes(Fa_N_down_kgpmps,0,1)
    for c in range(len(latitude)):
        Fa_N_top_swap[c] = Fa_N_top_kgpmps_swap[c] * timestep * 0.5 *(L_N_gridcell[c]+L_S_gridcell[c]) / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]
        Fa_N_down_swap[c] = Fa_N_down_kgpmps_swap[c] * timestep * 0.5*(L_N_gridcell[c]+L_S_gridcell[c]) / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]

    Fa_N_top = np.swapaxes(Fa_N_top_swap,0,1) 
    Fa_N_down = np.swapaxes(Fa_N_down_swap,0,1) 
    
    #find out where the negative fluxes are
    Fa_E_top_posneg = np.ones(np.shape(Fa_E_top))
    Fa_E_top_posneg[Fa_E_top < 0] = -1
    Fa_N_top_posneg = np.ones(np.shape(Fa_E_top))
    Fa_N_top_posneg[Fa_N_top < 0] = -1
    Fa_E_down_posneg = np.ones(np.shape(Fa_E_top))
    Fa_E_down_posneg[Fa_E_down < 0] = -1
    Fa_N_down_posneg = np.ones(np.shape(Fa_E_top))
    Fa_N_down_posneg[Fa_N_down < 0] = -1
    
    #make everything absolute   
    Fa_E_top_abs = np.abs(Fa_E_top)
    Fa_E_down_abs = np.abs(Fa_E_down)
    Fa_N_top_abs = np.abs(Fa_N_top)
    Fa_N_down_abs = np.abs(Fa_N_down)
    
    # stabilize the outfluxes / influxes
    stab = 1./2.  # during the reduced timestep the water cannot move further than 1/x * the gridcell, 
                    #in other words at least x * the reduced timestep is needed to cross a gridcell
    size=np.size(Fa_E_top_abs)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fa_E_top_stable = np.reshape(np.minimum(np.reshape(Fa_E_top_abs, size), (np.reshape(Fa_E_top_abs, size)  / 
                        (np.reshape(Fa_E_top_abs, size)  + np.reshape(Fa_N_top_abs, size))) * stab 
                         * np.reshape(W_top[:-1,:,:], size)),np.shape(Fa_E_top_abs))
    # with np.seterr(divide='ignore', invalid='ignore'):    
        Fa_N_top_stable = np.reshape(np.minimum(np.reshape(Fa_N_top_abs, size), (np.reshape(Fa_N_top_abs, size)  / 
                        (np.reshape(Fa_E_top_abs, size)  + np.reshape(Fa_N_top_abs, size))) * stab 
                         * np.reshape(W_top[:-1,:,:], size)),np.shape(Fa_N_top_abs))
    # with np.seterr(divide='ignore', invalid='ignore'):
        Fa_E_down_stable = np.reshape(np.minimum(np.reshape(Fa_E_down_abs, (size)), (np.reshape(Fa_E_down_abs, size)  / 
                        (np.reshape(Fa_E_down_abs, size)  + np.reshape(Fa_N_down_abs, size))) * stab 
                         * np.reshape(W_down[:-1,:,:], size)),np.shape(Fa_E_down_abs))
    # with np.seterr(divide='ignore', invalid='ignore'):
        Fa_N_down_stable = np.reshape(np.minimum(np.reshape(Fa_N_down_abs, size), (np.reshape(Fa_N_down_abs, size)  / 
                        (np.reshape(Fa_E_down_abs, size)  + np.reshape(Fa_N_down_abs, size))) * stab 
                         * np.reshape(W_down[:-1,:,:], size)),np.shape(Fa_N_down_abs))

    # Fa_E_top_stable = np.reshape(np.minimum(np.reshape(Fa_E_top_abs, size), stab* np.reshape(W_top[:-1,:,:], size)),np.shape(Fa_E_top_abs)
    # Fa_N_top_stable = np.reshape(np.minimum(np.reshape(Fa_N_top_abs, size), stab* np.reshape(W_top[:-1,:,:], size)),np.shape(Fa_N_top_abs))
    # Fa_E_down_stable = np.reshape(np.minimum(np.reshape(Fa_E_down_abs, (size)), stab* np.reshape(W_down[:-1,:,:], size)),np.shape(Fa_E_down_abs))
    # Fa_N_down_stable = np.reshape(np.minimum(np.reshape(Fa_N_down_abs, size), stab* np.reshape(W_down[:-1,:,:], size)),np.shape(Fa_N_down_abs))
    
    #get rid of the nan values
    Fa_E_top_stable[np.isnan(Fa_E_top_stable)] = 0
    Fa_N_top_stable[np.isnan(Fa_N_top_stable)] = 0
    Fa_E_down_stable[np.isnan(Fa_E_down_stable)] = 0
    Fa_N_down_stable[np.isnan(Fa_N_down_stable)] = 0

    #redefine
    Fa_E_top = Fa_E_top_stable * Fa_E_top_posneg
    Fa_N_top = Fa_N_top_stable * Fa_N_top_posneg
    Fa_E_down = Fa_E_down_stable * Fa_E_down_posneg
    Fa_N_down = Fa_N_down_stable * Fa_N_down_posneg
    
    return Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down

#%% Code
def getFa_Vert(Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down,E,P,W_top,W_down,time_reduce,latitude,longitude,isglobal):
    
    #total moisture in the column
    W = W_top + W_down
    
    #define the horizontal fluxes over the boundaries
    # fluxes over the eastern boundary
    Fa_E_top_boundary = np.nan*np.zeros(np.shape(Fa_E_top))
    Fa_E_top_boundary[:,:,:-1] = 0.5 * (Fa_E_top[:,:,:-1] + Fa_E_top[:,:,1:])
    if isglobal == 1:
        Fa_E_top_boundary[:,:,-1] = 0.5 * (Fa_E_top[:,:,-1] + Fa_E_top[:,:,0])
    Fa_E_down_boundary = np.nan*np.zeros(np.shape(Fa_E_down))
    Fa_E_down_boundary[:,:,:-1] = 0.5 * (Fa_E_down[:,:,:-1] + Fa_E_down[:,:,1:])
    if isglobal == 1:
        Fa_E_down_boundary[:,:,-1] = 0.5 * (Fa_E_down[:,:,-1] + Fa_E_down[:,:,0])

    # find out where the positive and negative fluxes are
    Fa_E_top_pos = np.ones(np.shape(Fa_E_top))
    Fa_E_down_pos = np.ones(np.shape(Fa_E_down))
    with np.errstate(invalid='ignore'):
        Fa_E_top_pos[Fa_E_top_boundary < 0] = 0
        Fa_E_down_pos[Fa_E_down_boundary < 0] = 0
    Fa_E_top_neg = Fa_E_top_pos - 1
    Fa_E_down_neg = Fa_E_down_pos - 1

    # separate directions west-east (all positive numbers)
    Fa_E_top_WE = Fa_E_top_boundary * Fa_E_top_pos
    Fa_E_top_EW = Fa_E_top_boundary * Fa_E_top_neg
    Fa_E_down_WE = Fa_E_down_boundary * Fa_E_down_pos
    Fa_E_down_EW = Fa_E_down_boundary * Fa_E_down_neg

    # fluxes over the western boundary
    Fa_W_top_WE = np.nan*np.zeros(np.shape(P))
    Fa_W_top_WE[:,:,1:] = Fa_E_top_WE[:,:,:-1]
    Fa_W_top_WE[:,:,0] = Fa_E_top_WE[:,:,-1]
    Fa_W_top_EW = np.nan*np.zeros(np.shape(P))
    Fa_W_top_EW[:,:,1:] = Fa_E_top_EW[:,:,:-1]
    Fa_W_top_EW[:,:,0] = Fa_E_top_EW[:,:,-1]
    Fa_W_down_WE = np.nan*np.zeros(np.shape(P))
    Fa_W_down_WE[:,:,1:] = Fa_E_down_WE[:,:,:-1]
    Fa_W_down_WE[:,:,0] = Fa_E_down_WE[:,:,-1]
    Fa_W_down_EW = np.nan*np.zeros(np.shape(P))
    Fa_W_down_EW[:,:,1:] = Fa_E_down_EW[:,:,:-1]
    Fa_W_down_EW[:,:,0] = Fa_E_down_EW[:,:,-1]    

    # fluxes over the northern boundary
    Fa_N_top_boundary = np.nan*np.zeros(np.shape(Fa_N_top))
    Fa_N_top_boundary[:,1:,:] = 0.5 * ( Fa_N_top[:,:-1,:] + Fa_N_top[:,1:,:] )
    Fa_N_down_boundary = np.nan*np.zeros(np.shape(Fa_N_down))
    Fa_N_down_boundary[:,1:,:] = 0.5 * ( Fa_N_down[:,:-1,:] + Fa_N_down[:,1:,:] )

    # find out where the positive and negative fluxes are
    Fa_N_top_pos = np.ones(np.shape(Fa_N_top))
    Fa_N_down_pos = np.ones(np.shape(Fa_N_down))
    with np.errstate(invalid='ignore'):
        Fa_N_top_pos[Fa_N_top_boundary < 0] = 0
        Fa_N_down_pos[Fa_N_down_boundary < 0] = 0
    Fa_N_top_neg = Fa_N_top_pos - 1
    Fa_N_down_neg = Fa_N_down_pos - 1

    # separate directions south-north (all positive numbers)
    Fa_N_top_SN = Fa_N_top_boundary * Fa_N_top_pos
    Fa_N_top_NS = Fa_N_top_boundary * Fa_N_top_neg
    Fa_N_down_SN = Fa_N_down_boundary * Fa_N_down_pos
    Fa_N_down_NS = Fa_N_down_boundary * Fa_N_down_neg

    # fluxes over the southern boundary
    Fa_S_top_SN = np.nan*np.zeros(np.shape(P))
    Fa_S_top_SN[:,:-1,:] = Fa_N_top_SN[:,1:,:]
    Fa_S_top_NS = np.nan*np.zeros(np.shape(P))
    Fa_S_top_NS[:,:-1,:] = Fa_N_top_NS[:,1:,:]
    Fa_S_down_SN = np.nan*np.zeros(np.shape(P))
    Fa_S_down_SN[:,:-1,:] = Fa_N_down_SN[:,1:,:]
    Fa_S_down_NS = np.nan*np.zeros(np.shape(P))
    Fa_S_down_NS[:,:-1,:] = Fa_N_down_NS[:,1:,:]

    # check the water balance
    Sa_after_Fa_down = np.nan*np.zeros([1,len(latitude),len(longitude)])
    Sa_after_Fa_top = np.nan*np.zeros([1,len(latitude),len(longitude)])
    #Sa_after_all_down = np.nan*np.zeros([1,len(latitude),len(longitude)])
    #Sa_after_all_top = np.nan*np.zeros([1,len(latitude),len(longitude)])
    Sa_after_all_down = np.nan*np.zeros(np.shape(P))
    Sa_after_all_top = np.nan*np.zeros(np.shape(P))
    residual_down = np.nan*np.zeros(np.shape(P)) # residual factor [m3]
    residual_top = np.nan*np.zeros(np.shape(P)) # residual factor [m3]
    
    # time_reduce=0.25
    len_time=np.int(24/time_reduce)
    for t in range(len_time):
        # down: calculate with moisture fluxes:
        Sa_after_Fa_down[0,1:-1,:] = (W_down[t,1:-1,:] - Fa_E_down_WE[t,1:-1,:] + Fa_E_down_EW[t,1:-1,:] + Fa_W_down_WE[t,1:-1,:] - Fa_W_down_EW[t,1:-1,:] - Fa_N_down_SN[t,1:-1,:] 
                                     + Fa_N_down_NS[t,1:-1,:] + Fa_S_down_SN[t,1:-1,:] - Fa_S_down_NS[t,1:-1,:])

        # top: calculate with moisture fluxes:
        Sa_after_Fa_top[0,1:-1,:] = (W_top[t,1:-1,:]- Fa_E_top_WE[t,1:-1,:] + Fa_E_top_EW[t,1:-1,:] + Fa_W_top_WE[t,1:-1,:] - Fa_W_top_EW[t,1:-1,:] - Fa_N_top_SN[t,1:-1,:] 
                                     + Fa_N_top_NS[t,1:-1,:] + Fa_S_top_SN[t,1:-1,:]- Fa_S_top_NS[t,1:-1,:])
    
        # down: substract precipitation and add evaporation
        Sa_after_all_down[t,1:-1,:] = Sa_after_Fa_down[0,1:-1,:] - P[t,1:-1,:] * (W_down[t,1:-1,:] / W[t,1:-1,:]) + E[t,1:-1,:]
    
        # top: substract precipitation
        Sa_after_all_top[t,1:-1,:] = Sa_after_Fa_top[0,1:-1,:] - P[t,1:-1,:] * (W_top[t,1:-1,:] / W[t,1:-1,:])
    
        # down: calculate the residual
        residual_down[t,1:-1,:] = W_down[t+1,1:-1,:] - Sa_after_all_down[t,1:-1,:]
    
        # top: calculate the residual
        residual_top[t,1:-1,:] = W_top[t+1,1:-1,:] - Sa_after_all_top[t,1:-1,:]

    # compute the resulting vertical moisture flux
    Fa_Vert_raw = W_down[:-1,:,:] / W[:-1,:,:] * (residual_down + residual_top) - residual_down # the vertical velocity so that the new residual_down/W_down =  residual_top/W_top (positive downward)
    # find out where the negative vertical flux is
    Fa_Vert_posneg = np.ones(np.shape(Fa_Vert_raw))
    with np.errstate(invalid='ignore'):
        Fa_Vert_posneg[Fa_Vert_raw < 0] = -1

    # make the vertical flux absolute
    Fa_Vert_abs = np.abs(Fa_Vert_raw)

    # stabilize the outfluxes / influxes
    stab = 1./4. #during the reduced timestep the vertical flux can maximally empty/fill 1/x of the top or down storage
    size=np.size(Fa_Vert_abs)
    Fa_Vert_stable = np.reshape(np.minimum(np.reshape(Fa_Vert_abs, size), np.minimum(stab*np.reshape(W_top[1:,:,:], size), 
                                        stab*np.reshape(W_down[1:,:,:], size))), np.shape(Fa_Vert_abs))
                
    # redefine the vertical flux
    Fa_Vert = Fa_Vert_stable * Fa_Vert_posneg
    #Fa_Vert = Fa_Vert_abs * Fa_Vert_posneg
    
     # Van der Ent, R. J., L. Wang-Erlandsson, P. W. Keys, and H. H. G. Savenije, Contrasting roles of interception and
    #transpiration in the hydrological cycle ¨C Part 2: Moisture recycling, Earth System Dynamics Discussions, 5, 281¨C
    #326, 2014.
    # Here based on the (B8) and (B7) in the above referrence, it should be Fa_Vert (positive upward)
    # Not Fa_Vert (positive downward) as stated in the above referrence
    # To verify,whether the residual meet the object in Eq. (B7) is tested. If you are interest, you can do it again
    '''
    residual_downs = np.nan*np.zeros(np.shape(residual_down)) # residual factor [m3]
    residual_tops = np.nan*np.zeros(np.shape(residual_down)) # residual factor [m3]
    residual_downs[:,1:-1,:] = W_down[1:,1:-1,:] - (Sa_after_all_down[:,1:-1,:]-Fa_Vert[:,1:-1,:])
    residual_tops[:,1:-1,:] = W_top[1:,1:-1,:] - (Sa_after_all_top[:,1:-1,:]+Fa_Vert[:,1:-1,:])
    test_obj=residual_downs/W_down[:-1,:,:]-residual_tops/W_top[:-1,:,:] # be zero if passing test
    '''

    return Fa_Vert
#%%
def data_path(yearnumber,a,input_folder):   
    q_f_data = os.path.join(input_folder, str(yearnumber) + '-Q-level.nc') #specific humidity 
    q_f_eoy_data = os.path.join(input_folder, str(yearnumber+1) + '-Q-level.nc')#specific humidity end of the year

    vint_data = os.path.join(input_folder, str(yearnumber) + '-Vintvar.nc') #total column water 
    vint_eoy_data = os.path.join(input_folder, str(yearnumber+1) + '-Vintvar.nc') #total column water end of the year

    u_f_data = os.path.join(input_folder, str(yearnumber) + '-U-level.nc' )
    u_f_eoy_data = os.path.join(input_folder, str(yearnumber+1) + '-U-level.nc' )

    v_f_data = os.path.join(input_folder, str(yearnumber) + '-V-level.nc' )
    v_f_eoy_data = os.path.join(input_folder, str(yearnumber+1) + '-V-level.nc' )

    evaporation_data = os.path.join(input_folder, str(yearnumber) + '-E.nc')
    precipitation_data = os.path.join(input_folder, str(yearnumber) + '-P.nc')    
    
    return q_f_data,q_f_eoy_data,vint_data,vint_eoy_data,u_f_data,u_f_eoy_data,v_f_data,v_f_eoy_data,evaporation_data,precipitation_data


#%% Main function for the Fluxes and States

def FLUX_States_mainfunction(yearnumber,a,time_reduce,latnrs,lonnrs,isglobal,input_folder,latitude,longitude,g,
    density_water,A_gridcell,L_N_gridcell,L_S_gridcell,L_EW_gridcell):
	# time_reduce: the reduced timestep, unit is hour
	# latnrs,lonnrs: manage the extent of your dataset, define the latitude and longitude cell numbers to consider
	# isglobal: fill in 1 for global computations (i.e. Earth round), fill in 0 for a local domain with boundaries
	
	# invariant_data = r'/public/home/mzxiao/ERA5/landseamask.nc' #invariants, land and sea mask
	ly = int(calendar.isleap(yearnumber))
	final_time = 364 + ly  # number of parts-1 to divide a year in
	# obtain the constants
	# latitude,longitude,lsm,g,density_water,A_gridcell,L_N_gridcell,L_S_gridcell,L_EW_gridcell = getconstants(latnrs,lonnrs,invariant_data)
	global datapath
	datapath=data_path(yearnumber,a,input_folder) # global variable

	dt_origin=datetime.datetime(1900,1,1)
	dt=datetime.datetime(yearnumber, 1, 1) + datetime.timedelta(days=np.float(a))
	dtt=dt-dt_origin
	time_low=dtt.days*24 # the boundary of time to select data
	time_up=time_low+23

    #1 integrate specific humidity to get the (total) column water (vapor)
	cw,W_top,W_down = getW(latnrs,lonnrs,final_time,a,time_low,time_up,density_water,latitude,longitude,g,A_gridcell)
            
	#2 wind in between pressure levels
	U,V = getwind(latnrs,lonnrs,final_time,a,time_low,time_up)
            
	#3 calculate horizontal moisture fluxes
	Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down = getFa(latnrs,lonnrs,cw,U,V,time_low,time_up,a,final_time)
            
	#4 evaporation and precipitation
	E,P = getEP(latnrs,lonnrs,time_low,time_up,latitude,longitude,A_gridcell)
            
	#5 put data on a smaller time step
	Fa_E_top_1,Fa_N_top_1,Fa_E_down_1,Fa_N_down_1,E,P,W_top,W_down = getrefined(Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,W_top,W_down,E,P,time_reduce,latitude,longitude)

	#6 stabilize horizontal fluxes and get everything in (m3 per smaller timestep)
	Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down = get_stablefluxes(W_top,W_down,Fa_E_top_1,Fa_E_down_1,Fa_N_top_1,Fa_N_down_1,
			time_reduce,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude,longitude)
            
	#7 determine the vertical moisture flux
	Fa_Vert= getFa_Vert(Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down,E,P,W_top,W_down,time_reduce,latitude,longitude,isglobal)
            
	return Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,Fa_Vert,E,P,W_top,W_down

