# -*- coding: utf-8 -*-
"""
Complex Trace Attributes for Seismic Data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
import util
from SignalProcess import SignalProcess as sp


class ComplexAttributes():
    """
    Description
    -----------
    Class object containing methods for computing complex trace attributes 
    from 3D seismic data.
    
    Methods
    -------
    create_array
    envelope
    instantaneous_phase 
    cosine_instantaneous_phase 
    relative_amplitude_change
    instantaneous_frequency 
    instantaneous_bandwidth
    dominant_frequency 
    frequency_change 
    sweetness 
    quality_factor 
    response_phase 
    response_frequency 
    response_amplitude 
    apparent_polarity
    """
        
    def create_array(self, darray, kernel=None, preview=None):
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
        
    
    def envelope(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Envelope of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        kernel = (1,1,25)
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        analytical_trace = darray.map_blocks(util.hilbert, dtype=darray.dtype)
        result = da.absolute(analytical_trace)
        result = util.trim_dask_array(result, kernel)
        
        return(result) 
        
    
    
    def instantaneous_phase(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Instantaneous Phase of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        kernel = (1,1,25)
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        analytical_trace = darray.map_blocks(util.hilbert, dtype=darray.dtype)
        result = da.rad2deg(da.angle(analytical_trace))
        result = util.trim_dask_array(result, kernel)
        
        return(result)
            
            
    def cosine_instantaneous_phase(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Cose of Instantaneous Phase of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, preview=preview)            
        phase = self.instantaneous_phase(darray)
        result = da.rad2deg(da.angle(phase))
        
        return(result)
            
    
    
    def relative_amplitude_change(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Relative Amplitude Change of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        env = self.envelope(darray)
        env_prime = sp().first_derivative(env, axis=-1)
        result = env_prime / env
        result = da.clip(result, -1, 1)
            
        return(result)
            
    
    
    def amplitude_acceleration(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Amplitude Acceleration of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, preview=preview)
        rac = self.relative_amplitude_change(darray)        
        result = sp().first_derivative(rac, axis=-1)
            
        return(result)

    
    
    def instantaneous_frequency(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Instantaneous Frequency of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        darray, chunks_init = self.create_array(darray, preview=preview)
        
        fs = 1000 / sample_rate
        phase = self.instantaneous_phase(darray)
        phase = da.deg2rad(phase)
        phase = phase.map_blocks(np.unwrap, dtype=darray.dtype)
        phase_prime = sp().first_derivative(phase, axis=-1)        
        result = da.absolute((phase_prime / (2.0 * np.pi) * fs))
                   
        return(result)
        
        
    def instantaneous_bandwidth(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Instantaneous Bandwidth of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------    
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, preview=preview)                  
        rac = self.relative_amplitude_change(darray)        
        result = da.absolute(rac) / (2.0 * np.pi)        
        
        return(result)
            
    
    def dominant_frequency(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Dominant Frequency of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        darray, chunks_init = self.create_array(darray, preview=preview)                    
        inst_freq = self.instantaneous_frequency(darray, sample_rate)
        inst_band = self.instantaneous_bandwidth(darray)        
        result = da.hypot(inst_freq, inst_band)
                    
        return(result)
        
        
    def frequency_change(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Frequency Change of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        darray, chunks_init = self.create_array(darray, preview=preview)
        inst_freq = self.instantaneous_frequency(darray, sample_rate)
        result = sp().first_derivative(inst_freq, axis=-1)
                    
        return(result)
        
        
    def sweetness(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Sweetness of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        def func(chunk):
            chunk[chunk < 5] = 5
            return(chunk)
        
        darray, chunks_init = self.create_array(darray, preview=preview)                    
        inst_freq = self.instantaneous_frequency(darray, sample_rate)
        inst_freq = inst_freq.map_blocks(func, dtype=darray.dtype)
        env = self.envelope(darray)
        
        result = env / inst_freq
                            
        return(result)
        
    
    def quality_factor(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Quality Factor of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        darray, chunks_init = self.create_array(darray, preview=preview)
                    
        inst_freq = self.instantaneous_frequency(darray, sample_rate)
        rac = self.relative_amplitude_change(darray)
        
        result = (np.pi * inst_freq) / rac
        
        return(result)       
        
        
    def response_phase(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Response Phase of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        def operation(chunk1, chunk2, chunk3):
            
            out = np.zeros(chunk1.shape)            
            for i,j in np.ndindex(out.shape[:-1]):
                
                ints = np.unique(chunk3[i, j, :])    
                
                for ii in ints:
                    
                    idx = np.where(chunk3[i, j, :] == ii)[0]
                    peak = idx[chunk1[i, j, idx].argmax()]
                    out[i, j, idx] = chunk2[i, j, peak]
                    
            return(out)
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        env = self.envelope(darray)
        phase = self.instantaneous_phase(darray)        
        troughs = env.map_blocks(util.local_events, comparator=np.less, 
                                 dtype=darray.dtype)
        troughs = troughs.cumsum(axis=-1)        
        result = da.map_blocks(operation, env, phase, troughs, dtype=darray.dtype)        
        result[da.isnan(result)] = 0
        
        
        return(result)
        
        
    def response_frequency(self, darray, sample_rate=4, preview=None):
        """
        Description
        -----------
        Compute the Response Frequency of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
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
        
        def operation(chunk1, chunk2, chunk3):
            
            out = np.zeros(chunk1.shape)            
            for i,j in np.ndindex(out.shape[:-1]):
                
                ints = np.unique(chunk3[i, j, :])    
                
                for ii in ints:
                    
                    idx = np.where(chunk3[i, j, :] == ii)[0]
                    peak = idx[chunk1[i, j, idx].argmax()]
                    out[i, j, idx] = chunk2[i, j, peak]
                    
            return(out)
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        env = self.envelope(darray)
        inst_freq = self.instantaneous_frequency(darray, sample_rate)
        troughs = env.map_blocks(util.local_events, comparator=np.less, 
                                 dtype=darray.dtype)
        troughs = troughs.cumsum(axis=-1)        
        result = da.map_blocks(operation, env, inst_freq, troughs, dtype=darray.dtype)        
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def response_amplitude(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Response Amplitude of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        def operation(chunk1, chunk2, chunk3):
            
            out = np.zeros(chunk1.shape)            
            for i,j in np.ndindex(out.shape[:-1]):
                
                ints = np.unique(chunk3[i, j, :])    
                
                for ii in ints:
                    
                    idx = np.where(chunk3[i, j, :] == ii)[0]
                    peak = idx[chunk1[i, j, idx].argmax()]
                    out[i, j, idx] = chunk2[i, j, peak]
                    
            return(out)
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        env = self.envelope(darray)
        troughs = env.map_blocks(util.local_events, comparator=np.less, 
                                 dtype=darray.dtype)
        troughs = troughs.cumsum(axis=-1)        
        result = da.map_blocks(operation, env, darray, troughs, dtype=darray.dtype)        
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def apparent_polarity(self, darray, preview=None):
        """
        Description
        -----------
        Compute the Apparent Polarity of the input data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        def operation(chunk1, chunk2, chunk3):
            
            out = np.zeros(chunk1.shape)            
            for i,j in np.ndindex(out.shape[:-1]):
                
                ints = np.unique(chunk3[i, j, :])    
                
                for ii in ints:
                    
                    idx = np.where(chunk3[i, j, :] == ii)[0]
                    peak = idx[chunk1[i, j, idx].argmax()]
                    out[i, j, idx] = chunk1[i, j, peak] * np.sign(chunk2[i, j, peak])
                    
            return(out)
                    
        darray, chunks_init = self.create_array(darray, preview=preview)        
        env = self.envelope(darray)
        troughs = env.map_blocks(util.local_events, comparator=np.less, 
                                 dtype=darray.dtype)
        troughs = troughs.cumsum(axis=-1)        
        result = da.map_blocks(operation, env, darray, troughs, dtype=darray.dtype)        
        result[da.isnan(result)] = 0
        
        return(result)
