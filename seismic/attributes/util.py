# -*- coding: utf-8 -*-
"""
Utilities for working with seismic and computing volume attributes.

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
import h5py
import psutil

# Ignore warning
import warnings
warnings.filterwarnings("ignore")

def compute_chunk_size(shape, byte_size, kernel=None, preview=None):
    """
    Description
    -----------
    Compute ideal block size for Dask Array given specific information about 
    the computer being used, the input data, kernel size, and whether or not
    this operation is is 'preview' mode.
    
    Parameters
    ----------
    shape : tuple (len 3), shape of seismic data
    byte_size : int, byte size of seismic data dtype
    
    Keywork Arguments
    -----------------    
    kernel : tuple (len 3), operator size
    preview : str, enables or disables preview mode and specifies direction
        Acceptable inputs are (None, 'inline', 'xline', 'z')
        Optimizes chunk size in different orientations to facilitate rapid
        screening of algorithm output
    
    Returns
    -------
    chunk_size : tuple (len 3), optimal chunk size
    """
    
    # Evaluate kernel
    if kernel == None:
        kernel = (1,1,1)
        ki, kj, kk = kernel
    else:
        ki, kj, kk = kernel
        
    # Identify acceptable chunk sizes
    i_s = np.arange(ki, shape[0])
    j_s = np.arange(kj, shape[1])
    k_s = np.arange(kk, shape[2])
    
    modi = shape[0] % i_s
    modj = shape[1] % j_s
    modk = shape[2] % k_s
    
    kki = i_s[(modi >= ki) | (modi == 0)]
    kkj = j_s[(modj >= kj) | (modj == 0)]
    kkk = k_s[(modk >= kk) | (modk == 0)]
    
    # Compute Machine Specific information
    mem = psutil.virtual_memory().available
    cpus = psutil.cpu_count()
    byte_size = byte_size     
    M = ((mem / (cpus * byte_size)) / (ki * kj * kk)) * 0.75
    
    # Compute chunk size if preview mode is disabled
    if preview == None:
#        M *= 0.3
        Mij = kki * kkj.reshape(-1,1) * shape[2]
        Mij[Mij > M] = -1
        Mij = Mij.diagonal()

        chunks = [kki[Mij.argmax()], kkj[Mij.argmax()], shape[2]]
    
    # Compute chunk size if preview mode is enabled
    else:        
        kki = kki.min()
        kkj = kkj.min()
        kkk = kkk.min()
        
        if preview == 'inline':
            if (kki * shape[1] * shape[2]) < M:
                chunks = [kki, shape[1], shape[2]]
            
            else:
                j_s = np.arange(kkj, shape[1])
                modj = shape[1] % j_s
                kkj = j_s[(modj >= kj) | (modj == 0)]
                Mj = j_s * kki * shape[2]
                Mj = Mj[Mj < M]
                chunks = [kki, Mj.argmax(), shape[2]]    
                
        elif preview == 'xline':
            if (kkj * shape[0] * shape[2]) < M:
                chunks = [shape[0], kkj, shape[2]]
                
            else:
                i_s = np.arange(kki, shape[0])
                modi = shape[0] % i_s
                kki = i_s[(modi >= ki) | (modi == 0)]
                Mi = i_s * kkj * shape[2]
                Mi = Mi[Mi < M]
                chunks = [Mi.argmax(), kkj, shape[2]]                    
            
        else:
            if (kkk * shape[0] * shape[1]) < M:
                chunks = [shape[0], shape[2], kk]
            
            else:
                j_s = np.arange(kkj, shape[1])
                modj = shape[1] % j_s
                kkj = j_s[(modj >= kj) | (modj == 0)]
                Mj = j_s * kkk * shape[0]
                Mj = Mj[Mj < M]
                chunks = [shape[0], Mj.argmax(), kkk]
        
    return(tuple(chunks))
    
        
        
def trim_dask_array(in_data, kernel):
    """
    Description
    -----------
    Trim resuling Dask Array given a specified kernel size
    
    Parameters
    ----------
    in_data : Dask Array
    kernel : tuple (len 3), operator size
    
    Returns
    -------
    out : Dask Array
    """
    
    # Compute half windows and assign to dict
    hw = tuple(np.array(kernel) // 2)    
    axes = {0 : hw[0], 1 : hw[1], 2: hw[2]}
    
    return(da.overlap.trim_internal(in_data, axes=axes))
    
    

def available_volumes(file_path):
    """
    Description
    -----------
    Convience function to evaluate what volumes exist and what their names are
    
    Parameters
    ----------
    file_path : str, path to file
       
    Returns
    -------
    vols : list, array of volume names in file
    """
    
    # Iterate through HDF5 file and output dataset names
    with h5py.File(file_path) as f:
        vols = [i for i in f]
    
    return(vols)
    


def read(file_path):
    """
    Description
    -----------
    Convience function to read file and create a pointer to data on disk
    
    Parameters
    ----------
    file_path : str, path to file
       
    Returns
    -------
    data : HDF5 dataset, pointer to data on disk
    """
    
    data = h5py.File(file_path)['data']
    
    return(data)
    
    

def save(out_data, out_file):
    """
    Description
    -----------
    Convience function to read file and create a pointer to data on disk
    
    Parameters
    ----------
    out_data : Dask Array, data to be saved to disk
    out_file : str, path to file to save to
    """
    
    # Save to disk if object is Dask Array
    try:       
        out_data.to_hdf5(out_file, 'data')
    except Exception:
        raise Exception('Object is not a Dask Array')
        
        

def convert_dtype(in_data, min_val, max_val, to_dtype):
    """
    Description
    -----------
    Convience function to read file and create a pointer to data on disk
    
    Parameters
    ----------
    in_data : Dask Array, data to convert
    min_val : number, lower clip
    max_val : number, upper clip
    to_dtype : NumPy dtype
        Acceptable formats include (np.int8, np.float16, np.float32)
       
    Returns
    -------
    out : Dask Array, converted data
    """
    
    # Check if data is already in correct format
    if in_data.dtype == to_dtype:
        return(in_data)
        
    
    else:              
        
        in_data = da.clip(in_data, min_val, max_val)        
        if to_dtype == np.int8:
            in_data = ((in_data - min_val) / (max_val - min_val))
            dtype = np.iinfo(np.int8)
            in_data *= dtype.max - dtype.min
            in_data -= dtype.min
            out = in_data.astype(np.int8)
        
        elif to_dtype == np.float16:
            out = in_data.astype(np.float16)
            
        elif to_dtype == np.int32:
            out = in_data.astype(np.float32)
            
        else:
            raise Exception('Not a valid dtype')
    
    return(out)
    
    
    
def extract_patches(in_data, kernel):
    """
    Description
    -----------
    Reshape in_data into a collection of patches defined by kernel
    
    Parameters
    ----------
    in_data : Dask Array, data to convert
    kernel : tuple (len 3), operator size
       
    Returns
    -------
    out : Numpy Array, has shape (in_data.shape[0], in_data.shape[1], 
                                  in_data.shape[2], kernel[0], kernel[1], kernel[2])
    """
    
    strides = in_data.strides + in_data.strides
    shape = (np.array(in_data.shape) - np.array(kernel)) + 1
    shape = tuple(list(shape) + list(kernel))
    
    patches = np.lib.stride_tricks.as_strided(in_data,
                                              shape=shape,
                                              strides=strides)        
    return(patches)
    
    

def local_events(in_data, comparator):
    """
    Description
    -----------
    Find local peaks or troughs depending on comparator used
    
    Parameters
    ----------
    in_data : Dask Array, data to convert
    comparator : function, defines truth between neighboring elements
       
    Returns
    -------
    out : Numpy Array
    """
    
    idx = np.arange(0, in_data.shape[-1])    
    trace = in_data.take(idx, axis=-1, mode='clip')
    plus = in_data.take(idx + 1, axis=-1, mode='clip')
    minus = in_data.take(idx - 1, axis=-1, mode='clip')
    
    result = np.ones(in_data.shape, dtype=np.bool)
    
    result &= comparator(trace, plus)
    result &= comparator(trace, minus)
    
    return(result)
    
    
    
def hilbert(in_data):
    N = in_data.shape[-1]
    Xf = np.fft.fft(in_data, n=N, axis=-1)

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if in_data.ndim > 1:
        # broadcast h across all leading axes
        ind = (None,) * (in_data.ndim - 1) + (slice(None),)
        h = h[ind]                 # tuple indexing works

        # (optional, equivalent)
        # h = h.reshape((1,) * (in_data.ndim - 1) + (N,))

    x = np.fft.ifft(Xf * h, axis=-1)
    return x
