# -*- coding: utf-8 -*-
"""
Frequency attributes for Seismic data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
from scipy import signal
import util



class Frequency():
    """
    Description
    -----------
    Class object containing methods for performing frequency filtering
    
    Methods
    -------
    create_array
    lowpass_filter
    highpass_filter
    bandpass_filter
    cwt_ricker
    cwt_ormsby
    """
    
    def create_array(self, darray, kernel, preview):
        """
        Description
        -----------
        Convert input to Dask Array with ideal chunk size as necessary.  Perform
        necessary ghosting as needed for opertations utilizing windowed functions.
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        kernel : tuple (len 3), operator size
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        darray : Dask Array
        chunk_init : tuple (len 3), chunk size before ghosting.  Used in select cases
        """
    
        # Compute chunk size and convert if not a Dask Array
        if not isinstance(darray, da.core.Array):  
            chunk_size = util.compute_chunk_size(darray.shape, 
                                               darray.dtype.itemsize, 
                                               kernel=kernel,
                                               preview=preview)
            darray = da.from_array(darray, chunks=chunk_size)
            chunks_init = darray.chunks            
                
        else:
            chunks_init = darray.chunks
        
        # Ghost Dask Array if operation specifies a kernel
        if kernel != None:
                hw = tuple(np.array(kernel) // 2)
                darray = da.overlap.overlap(darray, depth=hw, boundary='reflect')
                
        return(darray, chunks_init)
        
        
    def lowpass_filter(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform low pass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            
            out = signal.filtfilt(B, A, x=chunk)
            
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, freq/nyq, btype='lowpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
    def highpass_filter(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform high pass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            
            out = signal.filtfilt(B, A, x=chunk)
            
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, freq/nyq, btype='highpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
        
    def bandpass_filter(self, darray, freq_lp, freq_hp, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform bandpass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq_lp : Number (Hz), frequency cutoff used in low pass filter
        freq_hp : Number (Hz), frequency cutoff used in high pass filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            
            out = signal.filtfilt(B, A, x=chunk)
            
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, (freq_lp/nyq, freq_hp/nyq), btype='bandpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
        
    def cwt_ricker(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform Continuous Wavelet Transform using Ricker Wavelet
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency defining Ricker Wavelet
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Generate wavelet of specified frequency
        def wavelet(freq, sample_rate):
            
            sr = sample_rate / 1000            
            t = np.arange(-0.512 / 2, 0.512 / 2, sr)
            out = (1 - (2 * (np.pi * freq * t) ** 2)) * np.exp(-(np.pi * freq * t) ** 2)
            
            return(out)
            
        # Convolve wavelet with trace
        def convolve(chunk, w):
            
            out = np.zeros(chunk.shape)
            
            for i,j in np.ndindex(chunk.shape[:-1]):                
                out[i, j] = signal.fftconvolve(chunk[i, j], w, mode='same')
                
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)
        w = wavelet(freq, sample_rate)
        result = darray.map_blocks(convolve, w=w, dtype=darray.dtype)
        
        return(result)
        
        
    def cwt_ormsby(self, darray, freqs, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform Continuous Wavelet Transform using Ormsby Wavelet
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : tuple (len 4), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        # Generate wavelet of specified frequencyies
        def wavelet(freqs, sample_rate):
            
            f1, f2, f3, f4 = freqs
            sr = sample_rate / 1000            
            t = np.arange(-0.512 / 2, 0.512 / 2, sr)
            
            term1 = (((np.pi * f4) ** 2) / ((np.pi * f4) - (np.pi * f3))) * np.sinc(np.pi * f4 * t) ** 2
            term2 = (((np.pi * f3) ** 2) / ((np.pi * f4) - (np.pi * f3))) * np.sinc(np.pi * f3 * t) ** 2
            term3 = (((np.pi * f2) ** 2) / ((np.pi * f2) - (np.pi * f1))) * np.sinc(np.pi * f2 * t) ** 2
            term4 = (((np.pi * f1) ** 2) / ((np.pi * f2) - (np.pi * f1))) * np.sinc(np.pi * f1 * t) ** 2
            
            out = (term1 - term2) - (term3 - term4)
            
            return(out)
            
        # Convolve wavelet with trace
        def convolve(chunk, w):
            
            out = np.zeros(chunk.shape)
            
            for i,j in np.ndindex(chunk.shape[:-1]):                
                out[i, j] = signal.fftconvolve(chunk[i, j], w, mode='same')
                
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)
        w = wavelet(freqs, sample_rate)
        result = darray.map_blocks(convolve, w=w, dtype=darray.dtype)
        
        return(result)
