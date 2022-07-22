#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Writing and reading numpy file
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

# Save the data_arr variable into a .npy file
np.save('test_np_save.npy',data_arr)

data_arr_loaded = np.load('test_np_save.npy')
print(np.equal(data_arr,data_arr_loaded))
print(data_arr==data_arr_loaded)

#%%
"""
Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz',data_arr, data_arr2)

npzfile = np.load('test_savez.npz')
print(npzfile)
print('Variable names within this file:', sorted(npzfile.files))
print(npzfile['arr_0'])

print((data_arr==npzfile['arr_0']).all())  # apply the equal condition on all the the matrix and get one statement
print((data_arr2==npzfile['arr_1']).all())

#%%
"""
Error and exception
"""



#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = '/Users/aliaaafify/Documents/SWSSS/SWSSS_Data/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the discretization grid of the density data in 3D space. We will be using np.linspace to create evenly sapce data between the limits.

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

alt = 400
hi = np.where(altitudes_JB2008==alt) #generate an index 
"""
#create a canava wirh contour
fig, axs = plt.subplots(1, figsize=(15, 10*2))
ik =0
cs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
axs.set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
axs.set_ylabel("Latitudes", fontsize=18)
axs.tick_params(axis = 'both', which = 'major', labelsize = 16
                
# Make a colorbar for the ContourSet returned by the contourf call.
 cbar = fig.colorbar(cs,ax=axs[0])
 cbar.ax.set_ylabel('Density')
 axs.set_xlabel("Local Solar Time", fontsize=18)

"""
# Create a canvas to plot our data on. Here we are using a subplot with 5 spaces for the plots.
fig, axs = plt.subplots(3, figsize=(15, 10*2), sharex=True)

for ik in range (3):
     cs = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
     axs[ik].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
     axs[ik].set_ylabel("Latitudes", fontsize=18)
     axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
     
     # Make a colorbar for the ContourSet returned by the contourf call.
     cbar = fig.colorbar(cs,ax=axs[ik])
     cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)   

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

# Read the data from 
Density_JB2008=np.mean(np.mean(dens_data_feb1, axis=0),axis=0)

#uncertanity 
#eta=1e-16
#print()
#print()

# plot the density vs altitude
fig,ax = plt.subplots()
ax.plot( altitudes_JB2008,Density,'o', c='blue')
ax.set_xlabel('Altitude')
ax.set_ylabel('Density')
ax.grid(True)
plt.yscale('log')

#%%

JB2008_dens_feb1_alt=np.mean(np.mean(dens_data_feb1, axis=0),axis=0)
tiegcm_dens_feb1 = tiegcm_dens_reshaped[:,:,:,time_index]
tiegcm_dens_feb1_alt = np.mean(tiegcm_dens_feb1, axis=(0,1))
plt.subplots(1, figsize=(10, 6))
plt.semilogy(altitudes_tiegcm,tiegcm_dens_feb1_alt,linewidth = 2,label='TIE-GCM')
plt.semilogy(altitudes_JB2008,JB2008_dens_feb1_alt,'-.',linewidth = 2,label='JB2008')
plt.xlabel('Altitude', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.title('Mean Density vs Altitude', fontsize=18)
plt.grid()
plt.legend(fontsize=16)
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)

#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at 310km
"""
import h5py

load_data=h5py.File('/Users/aliaaafify/Documents/SWSSS/SWSSS_Data/TIEGCM/2002_TIEGCM_density.mat')
#%%
print('Key within the database:', list(load_data.keys()))

tiegcm_dens = (10**np.array(load_data["density"])*1000).T #convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(load_data['altitudes']).flatten()
latitudes_tiegcm = np.array(load_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(load_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]
time_array_tiegcm = np.linspace(0,8760,20, dtype = int)
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F') # Fortra-like index order

alt = 310
hi = np.where(altitudes_tiegcm==alt) #generate an index 

fig, axs = plt.subplots(3, figsize=(15, 10*2), sharex=True)

for ik in range (3):
     cs = axs[ik].contourf(localSolarTimes_tiegcm, latitudes_tiegcm, tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[ik]].squeeze().T)
     axs[ik].set_title('tiegcm density at 310 km, t = {} hrs'.format(time_array_tiegcm[ik]), fontsize=18)
     axs[ik].set_ylabel("Latitudes", fontsize=18)
     axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
     
     # Make a colorbar for the ContourSet returned by the contourf call.
     cbar = fig.colorbar(cs,ax=axs[ik])
     cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)   
#%%
#plotting density profile from tiegcm and JB2008

#Density_tiegcm = np.mean(load_data, axis=(0,1))
#Density_JB2008=np.mean(dens_data_feb1, axis=(0,1)
Density_tiegcm=np.mean(tiegcm_dens_reshaped, axis=(0,1))

# plot the density vs altitude
fig,ax = plt.subplots()
ax.plot( altitudes_JB2008,Density_JB2008, c='blue',label=JB2008)
ax.plot( altitudes_tiegcm,Density_tiegcm, c='orange',label=TIE-GCM)
ax.set_xlabel('Altitude')
ax.set_ylabel('Density')
ax.grid(True)
plt.yscale('log')
ax.legend()
#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)
interp_func_1D = interpolate.interp1d(x,y)
xnew = np.arange(0, 9, 0.1)
ynew = interp_func_1D(xnew)   # use interpolation function returned by `interp1d`

fig,ax = plt.subplots()
ax.plot( x,y,'o', c='blue')
ax.plot( xnew,ynew,'+', c='orange')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
#ax.legend()

from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolant on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)
interpolated_function_1 = RegularGridInterpolator((x, y, z), sample_data)
pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print(interpolated_function_1(pts))
print(function_1(pts[:,0],pts[:,1],pts[:,2]))
#%%
#3D interpolation


time_index = 31*24
JB2008_dens_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

JB2008dens_feb1_alt=np.mean(np.mean(JB2008_dens_feb1, axis=0),axis=0)
tiegcm_dens_feb1 = tiegcm_dens_reshaped[:,:,:,time_index]

xg, yg ,zg = np.meshgrid(localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm, indexing='ij', sparse=True)
sample_data = function_1(xg, yg, zg)

xgJ, ygJ ,zgJ = np.meshgrid(localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008, indexing='ij', sparse=True)
sample_data_JB2008 = function_1(xgJ, ygJ, zgJ)

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

alt= 400
hi=np.where(altitudes_JB2008==alt)

cs = axs[1].contourf(localSolarTimes_tiegcm , latitudes_tiegcm, tiegcm_grid.T)
axs[1].set_title('TIE-GCM density at 400 km, t = {} hrs'.format(time_index), fontsize=18)
axs[1].set_ylabel("Latitudes", fontsize=18)
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 16)

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs,ax=axs[1])
cbar.ax.set_ylabel('Density')
axs[ik].set_xlabel("Local Solar Time", fontsize=18


#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization grid.
Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on February 1st, 2002, with the discretized grid used for the JB2008 ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""

#import zipfile

#with zipfile.ZipFile('Data/jena_climate_2009_2016.csv.zip', 'r') as zip_ref:
#    zip_ref.extractall('Data/jena_climate_2009_2016/')


import pandas as pd

csv_path = '/Users/aliaaafify/Documents/SWSSS/SWSSS_Data/jena_climate_2009_2016.csv'
df = pd.read_csv(csv_path)
print(df)
df =df[5::6]
print(df)
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(date_time)
df.head()
df.tail()

plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)


df.describe().transpose()

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are then reflected in the DataFrame.
df['wv (m/s)'].min()

#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this difference in a contour plot.
"""





#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in terms of mean absolute percentage difference/error (MAPE). Let's plot the MAPE for this scenario.
"""





