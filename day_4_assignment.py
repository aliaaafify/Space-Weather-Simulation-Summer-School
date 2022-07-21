#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:03:29 2022

@author: aliaaafify
"""
# Import required packages
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse



def parse_args():
    """ using a parse to call the wanted variables """
    
    parser = argparse.ArgumentParser(description ='plotting TEC data from nc files') 
    parser.add_argument(
        'files_list',
        type=argparse.FileType('r'),
        nargs='+') #calling all files
    parser.add_argument(
        'dataset_dir',
        help='dataset_dir',
        type=argparse.FileType('r'),
        nargs='+') #calling the directory of the file
    parser.add_argument(
        '-output',
        help='figure into output file',
        type=str,default = 'tec.png',
        nargs='*') #saving the image
    
    args = parser.parse_args()
    print(args.list)
    return args


# start of the main code
if __name__ == '__main__' :
    #load the dataset for the TEC data
    args = parse_args()
    print(args.dataset_dir)  #print the directory where the database are
    for fi in args.files_list:
        
            dataset_arr[fi] = args.dataset_dir[fi] #the dataset array
            dataset[fi] = nc.Dataset(dataset_arr[fi]) #directory for the file
    
            dataset['tec'][:] #tec array
            dataset['lat'][:] #latitude array
            dataset['lon'][:] #longitutde array 
    
            TEC_unit = dataset['tec'].units # check the tec unit
            print(dataset) # show what is inside the dataset
    
            def plot_tec(dataset,figsize=(12,6)): #input 2 arrguments
                """Function to plot the TEC datat"""
        
                fig, ax = plt.subplots(1,figsize=figsize) #define the subplot figure from panels and sizes
                pmesh=ax.pcolormesh(dataset['lon'][:],dataset['lat'][:],dataset['tec'][:]) #put the 3 dataset on meshgrid
                ax.set_title("Plot of TEC of day 2022-07-20") #write down the title 
                ax.set_xlabel('longitude ('+ dataset['lon'].unit +')') #write dowm the x-axis label
                ax.set_ylabel('latitude (degree)') #write down the y-axis label
                fig.colorbar(pmesh, cax=None, ax=None) #set the colorbar for the figure
                fig.show() #figure show {it will not work for spyder}
                return fig,ax #that's the two variables that you need to insert
    
        fig, ax = plot_tec(dataset) #plot command for the function plot_tec
    
        outfile = args.filename +'.png'
        print('save as figure : ' + outfile) #this function will put my plot saved in the same directory as the code with the formated file name
        plt.savefig(outfile)
        plt.close()