# -*- coding: utf-8 -*-
"""
Dip & Azimuth Calculations for Seismic Data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import numpy as np
import dask.array as da
from scipy import ndimage as ndi
import util
from SignalProcess import SignalProcess as sp


class DipAzm():
    """
    Description
    -----------
    Class object containing methods for computing dip & azimuth attributes 
    from 3D seismic data.
    
    Methods
    -------
    create_array
    gradient_dips
    gradient_structure_tensor
    gst_2D_dips
    gst_3D_dip
    gst_3D_azm
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
        
        if not isinstance(darray, da.core.Array):  
            chunk_size = util.compute_chunk_size(darray.shape, 
                                               darray.dtype.itemsize, 
                                               kernel=kernel,
                                               preview=preview)
            darray = da.from_array(darray, chunks=chunk_size)
            chunks_init = darray.chunks            
                
        else:
            chunks_init = darray.chunks
        
        if kernel != None:
                hw = tuple(np.array(kernel) // 2)
                darray = da.overlap.overlap(darray, depth=hw, boundary='reflect')
                
        return(darray, chunks_init)
        
    
    def gradient_dips(self, darray, dip_factor=10, kernel=(3,3,3), preview=None):
        """
        Description
        -----------
        Compute Inline and Crossline Dip from the Inline, Crossline, & Z Gradients
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        dip_factor : Number, scalar for dip values
        kernel : tuple (len 3), operator size
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        il_dip : Dask Array, Inline Dips
        xl_dip : Dask Array, Crossline Dips
        """
        
        # Generate Dask Array as necessary
        darray, chunks_init = self.create_array(darray, kernel=None, preview=preview)
        
        # Compute I, J, K gradients
        gi = sp().first_derivative(darray, axis=0)
        gj = sp().first_derivative(darray, axis=1)
        gk = sp().first_derivative(darray, axis=2)
        
        # Compute dips
        il_dip = -(gi / gk) * dip_factor
        xl_dip = -(gj / gk) * dip_factor
                
        il_dip[da.isnan(il_dip)] = 0
        xl_dip[da.isnan(xl_dip)] = 0
        
        # Perform smoothing as specified
        if kernel != None:
            hw = tuple(np.array(kernel) // 2)
            il_dip = il_dip.map_overlap(ndi.median_filter, depth=hw, boundary='reflect', 
                                        dtype=darray.dtype, size=kernel)
            xl_dip = xl_dip.map_overlap(ndi.median_filter, depth=hw, boundary='reflect', 
                                        dtype=darray.dtype, size=kernel)               
        
        return(il_dip, xl_dip)
        

    def gradient_structure_tensor(self, darray, kernel, preview=None):
        """
        Description
        -----------
        Convience function to compute the Inner Product of Gradients
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        kernel : tuple (len 3), operator size
        
        Keywork Arguments
        -----------------  
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        gi2, gj2, gk2, gigj, gigk, gjgk : Dask Array
        """
        
        # Generate Dask Array as necessary
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)
        
        # Compute I, J, K gradients
        gi = sp().first_derivative(darray, axis=0)
        gj = sp().first_derivative(darray, axis=1)
        gk = sp().first_derivative(darray, axis=2)        
        gi = util.trim_dask_array(gi, kernel)
        gj = util.trim_dask_array(gj, kernel)
        gk = util.trim_dask_array(gk, kernel)
        
        # Compute Inner Product of Gradients
        hw = tuple(np.array(kernel) // 2)
        gi2 = (gi * gi).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                    dtype=darray.dtype, size=kernel)
        gj2 = (gj * gj).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                    dtype=darray.dtype, size=kernel)
        gk2 = (gk * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                    dtype=darray.dtype, size=kernel)
        gigj = (gi * gj).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                     dtype=darray.dtype, size=kernel)
        gigk = (gi * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                     dtype=darray.dtype, size=kernel)
        gjgk = (gj * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', 
                                     dtype=darray.dtype, size=kernel)
        
        return(gi2, gj2, gk2, gigj, gigk, gjgk)

    
    def gst_2D_dips(self, darray, dip_factor=10, kernel=(3,3,3), preview=None):
        """
        Description
        -----------
        Compute Inline and Crossline Dip from the Gradient Structure Tensor
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        dip_factor : Number, scalar for dip values
        kernel : tuple (len 3), operator size
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        il_dip : Dask Array, Inline Dips
        xl_dip : Dask Array, Crossline Dips
        """
        
        # Compute dips from Eigenvectors of GST
        def operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis):
            np.seterr(all='ignore')
            
            shape = gi2.shape
            
            gst = np.array([[gi2, gigj, gigk],
                          [gigj, gj2, gjgk],
                          [gigk, gjgk, gk2]])
            
            gst = np.moveaxis(gst, [0,1], [-2,-1])
            gst = gst.reshape((-1, 3, 3))
            
            evals, evecs = np.linalg.eigh(gst)
            ndx = evals.argsort()
            evecs = evecs[np.arange(0,gst.shape[0],1),:,ndx[:,2]]
            
            out = -evecs[:, axis] / evecs[:, 2]
            out = out.reshape(shape)
           
            return(out)
        
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.gradient_structure_tensor(darray, kernel, 
                                                                         preview=preview)        
        il_dip = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                               dtype=darray.dtype) 
        xl_dip = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, axis=1,
                               dtype=darray.dtype)
        
        il_dip *= dip_factor
        xl_dip *= dip_factor
        il_dip[da.isnan(il_dip)] = 0
        xl_dip[da.isnan(xl_dip)] = 0
        
        return(il_dip, xl_dip)
        
        
    def gst_3D_dip(self, darray, dip_factor=10, kernel=(3,3,3), preview=None):
        """
        Description
        -----------
        Compute 3D Dip from the Gradient Structure Tensor
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        dip_factor : Number, scalar for dip values
        kernel : tuple (len 3), operator size
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array, dip in degrees
        """
        
        # Function to compute 3D dip from GST
        def operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis):
            np.seterr(all='ignore')
            
            shape = gi2.shape
            
            gst = np.array([[gi2, gigj, gigk],
                          [gigj, gj2, gjgk],
                          [gigk, gjgk, gk2]])
            
            gst = np.moveaxis(gst, [0,1], [-2,-1])
            gst = gst.reshape((-1, 3, 3))
            
            evals, evecs = np.linalg.eigh(gst)
            ndx = evals.argsort()
            evecs = evecs[np.arange(0,gst.shape[0],1),:,ndx[:,2]]
            
            norm_factor = np.linalg.norm(evecs, axis = -1)
            evecs[:, 0] /= norm_factor
            evecs[:, 1] /= norm_factor
            evecs[:, 2] /= norm_factor
            
            evecs[evecs[:, 2] < 0] *= np.sign(evecs[evecs[:, 2] < 0])
            
            dip = np.dot(evecs, np.array([0,0,1]))
            dip = np.arccos(dip)
            dip = dip.reshape(shape)
            
            dip = np.rad2deg(dip)# - 90
            
            return(dip)
        
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.gradient_structure_tensor(darray, kernel, 
                                                                         preview=preview)        
        result = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                               dtype=darray.dtype) 
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def gst_3D_azm(self, darray, dip_factor=10, kernel=(3,3,3), preview=None):
        """
        Description
        -----------
        Compute 3D Azimuth from the Gradient Structure Tensor
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        
        Keywork Arguments
        -----------------  
        dip_factor : Number, scalar for dip values
        kernel : tuple (len 3), operator size
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array, azimuth in degrees
        """
        
        # Function to compute 3D azimuth from GST
        def operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis):
            np.seterr(all='ignore')
            
            shape = gi2.shape
            
            gst = np.array([[gi2, gigj, gigk],
                          [gigj, gj2, gjgk],
                          [gigk, gjgk, gk2]])
            
            gst = np.moveaxis(gst, [0,1], [-2,-1])
            gst = gst.reshape((-1, 3, 3))
            
            evals, evecs = np.linalg.eigh(gst)
            ndx = evals.argsort()
            evecs = evecs[np.arange(0,gst.shape[0],1),:,ndx[:,2]]
            
            norm_factor = np.linalg.norm(evecs, axis = -1)
            evecs[:, 0] /= norm_factor
            evecs[:, 1] /= norm_factor
            evecs[:, 2] /= norm_factor
            
            evecs[evecs[:, 2] < 0] *= np.sign(evecs[evecs[:, 2] < 0])
            
            azm = np.arctan2(evecs[:, 0], evecs[:, 1])
            azm = azm.reshape(shape)
            azm = np.rad2deg(azm)
            azm[azm < 0] += 360
            
            return(azm)
        
        # Compute Inner Product of Gradients and Azimuth
        gi2, gj2, gk2, gigj, gigk, gjgk = self.gradient_structure_tensor(darray, kernel, 
                                                                         preview=preview)        
        result = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                               dtype=darray.dtype) 
        result[da.isnan(result)] = 0
        
        return(result)
        
