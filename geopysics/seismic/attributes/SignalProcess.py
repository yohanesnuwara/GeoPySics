# -*- coding: utf-8 -*-
"""
Various signal processing attributes for Seismic Data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from scipy import signal
import util



class SignalProcess():
    """
    Description
    -----------
    Class object containing methods for performing various Signal Processing
    algorithms
    
    Methods
    -------
    create_array
    first_derivative
    second_derivative
    histogram_equalization
    time_gain
    rescale_amplitude_range
    rms
    trace_agc
    gradient_magnitude
    reflection_intensity
    phase_rotation
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
        
    
    def first_derivative(self, darray, axis=-1, preview=None):
        """
        Description
        -----------
        Compute first derivative of seismic data in specified direction
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        axis : Number, axis dimension
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        kernel = (3,3,3)
        axes = [ax for ax in range(darray.ndim) if ax != axis]
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)        
        result0 = darray.map_blocks(ndi.correlate1d, weights=[-0.5, 0, 0.5], 
                                   axis=axis, dtype=darray.dtype)
        result1 = result0.map_blocks(ndi.correlate1d, weights=[0.178947,0.642105,0.178947], 
                                   axis=axes[0], dtype=darray.dtype)
        result2 = result1.map_blocks(ndi.correlate1d, weights=[0.178947,0.642105,0.178947], 
                                     axis=axes[1], dtype=darray.dtype)
        result = util.trim_dask_array(result2, kernel)
        
        return(result)
        
        
    def second_derivative(self, darray, axis=-1, preview=None):
        """
        Description
        -----------
        Compute second derivative of seismic data in specified direction
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        axis : Number, axis dimension
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        kernel = (5,5,5)
        axes = [ax for ax in range(darray.ndim) if ax != axis]
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)        
        result0 = darray.map_blocks(ndi.correlate1d, weights=[0.232905, 0.002668, -0.471147, 0.002668, 0.232905], 
                                    axis=axis, dtype=darray.dtype)
        result1 = result0.map_blocks(ndi.correlate1d, weights=[0.030320, 0.249724, 0.439911, 0.249724, 0.030320], 
                                   axis=axes[0], dtype=darray.dtype)
        result2 = result1.map_blocks(ndi.correlate1d, weights=[0.030320, 0.249724, 0.439911, 0.249724, 0.030320], 
                                   axis=axes[1], dtype=darray.dtype)
        result = util.trim_dask_array(result2, kernel)
        
        return(result)
        
    
    def histogram_equalization(self, darray, preview=None):
        """
        Description
        -----------
        Perform histogram equalization of seismic data
        
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
        
        # Function to interpolate seismic to new scaling
        def interp(chunk, cdf, bins):
            
            out = np.interp(chunk.ravel(), bins, cdf)
            
            return(out.reshape(chunk.shape))
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        hist, bins = da.histogram(darray, bins=np.linspace(darray.min(), darray.max(), 
                                                           256, dtype=darray.dtype))
        cdf = hist.cumsum(axis=-1)
        cdf = cdf / cdf[-1]
        bins = (bins[:-1] + bins[1:]) / 2
        
        result = darray.map_blocks(interp, cdf=cdf, bins=bins, dtype=darray.dtype)
        
        return(result)
        
        
    def time_gain(self, darray, gain_val=1.5, preview=None):
        """
        Description
        -----------
        Gain the amplitudes in the Z/K dimension
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        gain_val : Float, exponential value
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, preview=preview)        
        z_ind = da.ones(darray.shape, chunks=darray.chunks).cumsum(axis=-1)
        gain = (1 + z_ind) ** gain_val
        
        result = darray * gain
        
        return(result)
        
        
       
    def rescale_amplitude_range(self, darray, min_val, max_val, preview=None):
        """
        Description
        -----------
        Clip the seismic data to specified values
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        min_val : Number, min clip value
        max_val : Numer, max clip value
        
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
        result = da.clip(darray, min_val, max_val)
        
        return(result)
        
        
    def rms(self, darray, kernel=(1,1,9), preview=None):
        """
        Description
        -----------
        Compute the Root Mean Squared (RMS) value within a specified window
        
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
        result : Dask Array
        """
        
        # Function to extract patches and perform algorithm 
        def operation(chunk, kernel):
            x = util.extract_patches(chunk, kernel)
            out = np.sqrt(np.mean(x ** 2, axis=(-3, -2, -1)))

            return(out)
        
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        result = darray.map_blocks(operation, kernel=kernel, dtype=darray.dtype, chunks=darray.chunks)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0 
        
        return(result)
        
        
    def trace_agc(self, darray, kernel=(1,1,9), preview=None):
        """
        Description
        -----------
        Apply an adaptive trace gain to input seismic
        
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
        result : Dask Array
        """
        
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)        
        rms = self.rms(darray, kernel)
        rms_max = rms.max()
        result = darray * (1.5 - (rms / rms_max))
        result[da.isnan(result)] = 0 
        
        return(result)
        
        
    def gradient_magnitude(self, darray, sigmas=(1,1,1), preview=None):
        """
        Description
        -----------
        Compute the 3D Gradient Magnitude using a Gaussian Operator
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        sigmas : tuple (len 3), gaussian operator in I, J, K
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        kernel = tuple(2 * (4 * np.array(sigmas) + 0.5).astype(int) + 1)
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        result = darray.map_blocks(ndi.gaussian_gradient_magnitude, sigma=sigmas, dtype=darray.dtype)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def reflection_intensity(self, darray, kernel=(1,1,9), preview=None):
        """
        Description
        -----------
        Compute reflection intensity by integrating the trace over a specified window
        
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
        result : Dask Array
        """
        
        # Function to extract patches and perform algorithm
        def operation(chunk, kernel):
            x = util.extract_patches(chunk, (1, 1, kernel[-1]))
            out = np.trapz(x).reshape(x.shape[:3])    
            
            return(out)
        
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        result = darray.map_blocks(operation, kernel=kernel, dtype=darray.dtype, chunks=chunks_init)
        result[da.isnan(result)] = 0 
        
        return(result)
        
    
    def phase_rotation(self, darray, rotation, preview=None):
        """
        Description
        -----------
        Rotate the phase of the seismic data by a specified angle
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        rotation : Number (degrees), angle of rotation
        
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
        
        phi = np.deg2rad(rotation)
        kernel = (1,1,25)
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        analytical_trace = darray.map_blocks(signal.hilbert, dtype=darray.dtype)
        result = analytical_trace.real * da.cos(phi) - analytical_trace.imag * da.sin(phi)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0
        
        return(result)
