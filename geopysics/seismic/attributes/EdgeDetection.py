# -*- coding: utf-8 -*-
"""
Edge Detection Attributes for Seismic Data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries

import dask.array as da
import numpy as np
from scipy import ndimage as ndi
import util
from SignalProcess import SignalProcess as sp



class EdgeDetection():
    """
    Description
    -----------
    Class object containing methods for computing edge attributes 
    from 3D seismic data.
    
    Methods
    -------
    create_array
    semblance
    eig_complex
    gradient_structure_tensor
    chaos
    volume_curvature
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


    def semblance(self, darray, kernel=(3,3,9), preview=None):
        """
        Description
        -----------
        Compute multi-trace semblance from 3D seismic
        
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
            np.seterr(all='ignore')
            x = util.extract_patches(chunk, kernel)
            s1 = np.sum(x, axis=(-3,-2)) ** 2
            s2 = np.sum(x ** 2, axis=(-3,-2))
            sembl = s1.sum(axis = -1) / s2.sum(axis = -1)
            sembl /= kernel[0] * kernel[1]

            return(sembl)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel, preview)                
        result = darray.map_blocks(operation, kernel=kernel, dtype=darray.dtype, chunks=chunks_init)
        result[da.isnan(result)] = 0 
        
        return(result)
        
        
        
    def gradient_structure_tensor(self, darray, kernel=(3,3,9), preview=None):
        """
        Description
        -----------
        Compute discontinuity from eigenvalues of gradient structure tensor
        
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
        def operation(gi2, gj2, gk2, gigj, gigk, gjgk):
            np.seterr(all='ignore')
            
            chunk_shape = gi2.shape
            
            gst = np.array([[gi2, gigj, gigk],
                          [gigj, gj2, gjgk],
                          [gigk, gjgk, gk2]])
            
            gst = np.moveaxis(gst, [0,1], [-2,-1])
            gst = gst.reshape((-1, 3, 3))
            
            eigs = np.sorgst(np.linalg.eigvalsh(gst))
            e1 = eigs[:, 2].reshape(chunk_shape)
            e2 = eigs[:, 1].reshape(chunk_shape)
            e3 = eigs[:, 0].reshape(chunk_shape)
                
            # Compute cvals from Eigenvalues
            cline = (e2 - e3) / (e2 + e3)
            cplane = (e1 - e2) / (e1 + e2)
            cfault = cline * (1 - cplane)
            
            return(cfault)
        
        # Generate Dask Array as necessary 
        darray, chunks_init = self.create_array(darray, kernel, preview)            
        
        # Compute I, J, K gradients
        gi = sp().first_derivative(darray, axis=0)
        gj = sp().first_derivative(darray, axis=1)
        gk = sp().first_derivative(darray, axis=2)
        
        # Compute the Inner Product of the Gradients
        gi2 = (gi * gi).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gj2 = (gj * gj).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gk2 = (gk * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gigj = (gi * gj).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gigk = (gi * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gjgk = (gj * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
            
        result = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, 
                               dtype=darray.dtype)    
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def eig_complex(self, darray, kernel=(3,3,9), preview=None):
        """
        Description
        -----------
        Compute multi-trace semblance from 3D seismic incorporating the 
            analytic trace
        
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
        
        # Function to compute the COV
        def cov(x, ki, kj, kk):
            x = x.reshape((ki * kj, kk))
            x = np.hstack([x.real, x.imag])
            return(x.dot(x.T))
        
        # Function to extract patches and perform algorithm 
        def operation(chunk, kernel):
            np.seterr(all='ignore')
            ki, kj, kk = kernel
            patches = util.extract_patches(chunk, kernel)
            
            out_data = []
            for i in range(0, patches.shape[0]):
                traces = patches[i]
                traces = traces.reshape(-1, ki * kj * kk)
                cov = np.apply_along_axis(cov, 1, traces, ki, kj, kk)
                vals = np.linalg.eigvals(cov)
                vals = np.abs(vals.max(axis=1) / vals.sum(axis=1))
            
                out_data.append(vals)
            
            out_data = np.asarray(out_data).reshape(patches.shape[:3])            
            
            return(out_data)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel, preview)        
        hilbert = darray.map_blocks(util.hilbert, dtype=darray.dtype)
        result = hilbert.map_blocks(operation, kernel=kernel, dtype=darray.dtype)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def chaos(self, darray, kernel=(3,3,9), preview=None):
        """
        Description
        -----------
        Compute multi-trace chaos from 3D seismic
        
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
        def operation(gi2, gj2, gk2, gigj, gigk, gjgk):
            np.seterr(all='ignore')
            
            chunk_shape = gi2.shape
            
            gst = np.array([[gi2, gigj, gigk],
                          [gigj, gj2, gjgk],
                          [gigk, gjgk, gk2]])
            
            gst = np.moveaxis(gst, [0,1], [-2,-1])
            gst = gst.reshape((-1, 3, 3))
            
            eigs = np.sort(np.linalg.eigvalsh(gst))
            e1 = eigs[:, 2].reshape(chunk_shape)
            e2 = eigs[:, 1].reshape(chunk_shape)
            e3 = eigs[:, 0].reshape(chunk_shape)
                
            out = (2 * e2) / (e1 + e3)
            
            return(out)
        
        # Generate Dask Array as necessary 
        darray, chunks_init = self.create_array(darray, kernel, preview)           
        
        # Compute I, J, K gradients
        gi = sp().first_derivative(darray, axis=0)
        gj = sp().first_derivative(darray, axis=1)
        gk = sp().first_derivative(darray, axis=2)
        
        # Compute the Inner Product of the Gradients
        gi2 = (gi * gi).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gj2 = (gj * gj).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gk2 = (gk * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gigj = (gi * gj).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gigk = (gi * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
        gjgk = (gj * gk).map_blocks(ndi.uniform_filter, size=kernel, dtype=darray.dtype)
            
        result = da.map_blocks(operation, gi2, gj2, gk2, gigj, gigk, gjgk, 
                               dtype=darray.dtype)    
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0
        
        return(result)
        
        
    def volume_curvature(self, darray_il, darray_xl, dip_factor=10, kernel=(3,3,3), 
                         preview=None):
        """
        Description
        -----------
        Compute volume curvature attributes from 3D seismic dips
        
        Parameters
        ----------
        darray_il : Array-like, Inline dip - acceptable inputs include 
            Numpy, HDF5, or Dask Arrays
        darray_xl : Array-like, Crossline dip - acceptable inputs include 
            Numpy, HDF5, or Dask Arrays
        
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
        H, K, Kmax, Kmin, KMPos, KMNeg : Dask Array, {H : 'Mean Curvature', 
                                                      K : 'Gaussian Curvature',
                                                      Kmax : 'Max Curvature',
                                                      Kmin : 'Min Curvature',
                                                      KMPos : Most Positive Curvature,
                                                      KMNeg : Most Negative Curvature}
        """
        
        np.seterr(all='ignore')
        
        # Generate Dask Array as necessary
        darray_il, chunks_init = self.create_array(darray_il, kernel, preview=preview)
        darray_xl, chunks_init = self.create_array(darray_xl, kernel, preview=preview)
        
        u = -darray_il / dip_factor
        v = -darray_xl / dip_factor
        w = da.ones_like(u, chunks=u.chunks)
        
        # Compute Gradients
        ux = sp().first_derivative(u, axis=0)
        uy = sp().first_derivative(u, axis=1)
        uz = sp().first_derivative(u, axis=2)
        vx = sp().first_derivative(v, axis=0)
        vy = sp().first_derivative(v, axis=1)
        vz = sp().first_derivative(v, axis=2)
        
        # Smooth Gradients
        ux = ux.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        uy = uy.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        uz = uz.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        vx = vx.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        vy = vy.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        vz = vz.map_blocks(ndi.uniform_filter, size=kernel, dtype=ux.dtype)
        
        u = util.trim_dask_array(u, kernel) 
        v = util.trim_dask_array(v, kernel) 
        w = util.trim_dask_array(w, kernel) 
        ux = util.trim_dask_array(ux, kernel) 
        uy = util.trim_dask_array(uy, kernel) 
        uz = util.trim_dask_array(uz, kernel)
        vx = util.trim_dask_array(vx, kernel)
        vy = util.trim_dask_array(vy, kernel)
        vz = util.trim_dask_array(vz, kernel)
        
        wx = da.zeros_like(ux, chunks=ux.chunks, dtype=ux.dtype)
        wy = da.zeros_like(ux, chunks=ux.chunks, dtype=ux.dtype)
        wz = da.zeros_like(ux, chunks=ux.chunks, dtype=ux.dtype)
        
        uv = u * v
        vw = v * w
        u2 = u * u
        v2 = v * v
        w2 = w * w
        u2pv2 = u2 + v2
        v2pw2 = v2 + w2
        s = da.sqrt(u2pv2 + w2)
        
        # Measures of surfaces
        E = da.ones_like(u, chunks=u.chunks, dtype=u.dtype)
        F = -(u * w) / (da.sqrt(u2pv2) * da.sqrt(v2pw2))
        G = da.ones_like(u, chunks=u.chunks, dtype=u.dtype)
        D = -(-uv * vx+u2 * vy + v2 * ux - uv * uy) / (u2pv2 * s)
        Di = -(vw * (uy + vx) - 2 * u * w * vy - v2 * (uz + wx) + uv * (vz + wy)) / (2 * da.sqrt(u2pv2) * da.sqrt(v2pw2) * s)
        Dii = -(-vw * wy + v2 * wz + w2 * vy - vw * vz) / (v2pw2 *s)
        H = (E * Dii - 2 * F *Di + G * D) / (2 * (E * G - F * F))
        K = (D * Dii - Di * Di) / (E * G - F * F)
        Kmin = H - da.sqrt(H * H - K)
        Kmax = H + da.sqrt(H * H - K)
        
        H[da.isnan(H)] = 0
        K[da.isnan(K)] = 0
        Kmax[da.isnan(Kmax)] = 0
        Kmin[da.isnan(Kmin)] = 0
        
        KMPos = da.maximum(Kmax, Kmin)
        KMNeg = da.minimum(Kmax, Kmin)
        
        return(H, K, Kmax, Kmin, KMPos, KMNeg)
