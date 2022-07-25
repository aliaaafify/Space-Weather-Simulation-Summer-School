#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:26:05 2022

@author: aliaaafify
"""
# Import required packages
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import argparse
import h5py
import matplotlib.pyplot as plt

# start of the main code
if __name__ == '__main__' :
    
        
    # Load Density Data for Jb2008 and TIEGCM
    dir_density_Jb2008 =  '/Users/aliaaafify/Documents/SWSSS/SWSSS_Data/JB2008/2002_JB2008_density.mat'
    alt = 400
    time_index = 31*24
    
    try:
        load_data_Jb2008 = loadmat(dir_density_Jb2008)
        print (load_data_Jb2008)
    except:
        print("File not found. Please check your directory")
    
    load_data_tiegcm = ''
    print(load_data_tiegcm)
    alt=args.alt
    time_index=args.time_index
    dir_density_TIEGCM = h5py.File(load_data_tiegcm)
    tiegcm_dens = (10**np.array(dir_density_TIEGCM["density"])*1000).T #convert from g/cm3 to kg/m3
    JB2008_dens = load_data_Jb2008['densityData']
    
    # Uses key to extract our data of interest for data model of JB2008
    localSolarTimes_JB2008 = np.linspace(0,24,24)
    latitudes_JB2008 = np.linspace(-87.5,87.5,20)
    altitudes_JB2008 = np.linspace(100,800,36)
    nofAlt_JB2008 = altitudes_JB2008.shape[0]
    nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
    nofLat_JB2008 = latitudes_JB2008.shape[0]
    
    # We can also impose additional constratints such as forcing the values to be integers.
    time_array_JB2008 = np.linspace(0,8760,20, dtype = int)
    
    # For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
    JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortra-like index order
    hi = np.where(altitudes_JB2008==alt) #generate an index 
    
    
    # extract our data of interest for data model of tiegcm
    altitudes_tiegcm = np.array(dir_density_TIEGCM['altitudes']).flatten()
    latitudes_tiegcm = np.array(dir_density_TIEGCM['latitudes']).flatten()
    localSolarTimes_tiegcm = np.array(dir_density_TIEGCM['localSolarTimes']).flatten()
    nofAlt_tiegcm = altitudes_tiegcm.shape[0]
    nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
    nofLat_tiegcm = latitudes_tiegcm.shape[0]
    time_array_tiegcm = np.linspace(0,8760,20, dtype = int)
    tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F') # Fortra-like index order
    
    hi = np.where(altitudes_tiegcm==alt) #generate an index 
    
    fig, axs = plt.subplots(2, figsize=(15, 10*2), sharex=True)
    
    for ik in range (2):
         cs = axs[ik].contourf(localSolarTimes_tiegcm, latitudes_tiegcm, tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[ik]].squeeze().T)
         axs[ik].set_title('tiegcm density at 310 km, t = {} hrs'.format(time_array_tiegcm[ik]), fontsize=18)
         axs[ik].set_ylabel("Latitudes", fontsize=18)
         axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
         
         # Make a colorbar for the ContourSet returned by the contourf call.
         cbar = fig.colorbar(cs,ax=axs[ik])
         cbar.ax.set_ylabel('Density')
    
    axs[ik].set_xlabel("Local Solar Time", fontsize=18)   
    
    JB2008_dens_feb1 = JB2008_dens_reshaped[:,:,:,time_index]
    
    JB2008dens_feb1_alt=np.mean(np.mean(JB2008_dens_feb1, axis=0),axis=0)
    tiegcm_dens_feb1 = tiegcm_dens_reshaped[:,:,:,time_index]
    
    
    tiegcm_function = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm), tiegcm_dens_feb1)
    JB2008_function = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008), JB2008_dens_feb1)
    
    print('Tie-gcm density at (lst=20hours, lat=12deg and alt=400 km)' , tiegcm_function((20.2,12,400)))
    
    lat3d = np.linspace(-90,90,180)
    time3d = np.linspace(0,24,180)
    
    tiegcm_grid = np.zeros((len(localSolarTimes_tiegcm),len(latitudes_tiegcm)))
    JB2008_grid = np.zeros((len(localSolarTimes_JB2008),len(latitudes_JB2008)))
    
    for lst_i in range (len(localSolarTimes_tiegcm)):
        for lat_i in range (len(latitudes_tiegcm)):
            tiegcm_grid[lst_i,lat_i] = tiegcm_function((localSolarTimes_tiegcm[lst_i],latitudes_tiegcm[lat_i],400))
            
    for lst_i in range (len(localSolarTimes_JB2008)):
        for lat_i in range (len(latitudes_JB2008)):
            JB2008_grid[lst_i,lat_i] = JB2008_function((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],400))
            
    fig, axs = plt.subplots(2, figsize=(15, 10), sharex=True)
    cs = axs[0].contourf(localSolarTimes_JB2008 , latitudes_JB2008, JB2008_grid.T)
    axs[0].set_title('TIE-JB2008 density at 310 km, t = {} hrs'.format(time_index), fontsize=18)
    axs[0].set_ylabel("Latitudes", fontsize=18)
    cbar = fig.colorbar(cs,ax=axs[0])
    cbar.ax.set_ylabel('Density')
    axs[0].set_xlabel("Local Solar Time", fontsize=18)
    
    
    cs = axs[1].contourf(localSolarTimes_tiegcm , latitudes_tiegcm, tiegcm_grid.T)
    axs[1].set_title('TIE-GCM density at 400 km, t = {} hrs'.format(time_index), fontsize=18)
    axs[1].set_ylabel("Latitudes", fontsize=18)
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[1])
    cbar.ax.set_ylabel('Density')
    axs[ik].set_xlabel("Local Solar Time", fontsize=18)
    