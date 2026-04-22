# -*- coding: utf-8 -*-
"""
Utilities for reading seismic data from SEGY and converting to usable format.

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
import segyio
import h5py
from shutil import copyfile as cf


def segy_read(segy_path, out_path, out_name):
    
    def write(chunk, segy_file, dset):
        for i in chunk:
            dset[i[0], i[1], :] = segy_file.trace.raw[i[2]]
            
        return(chunk)
    
    segy_file = segyio.open(segy_path)
    trace_inlines = segy_file.attributes(segyio.TraceField.INLINE_3D)[:]
    trace_xlines = segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    
    trace_inlines_unique = np.unique(trace_inlines)
    trace_xlines_unique = np.unique(trace_xlines)
    
    num_inline = trace_inlines_unique.size
    num_xline = trace_xlines_unique.size
    num_zsamples = len(segy_file.samples)
    
    min_inline = trace_inlines_unique.min()
    min_xline = trace_xlines_unique.min()
    min_zsample = segy_file.samples.min()
    
    max_inline = trace_inlines_unique.max()
    max_xline = trace_xlines_unique.max()
    max_zsample = segy_file.samples.max()
    
    inc_inline = int((max_inline - min_inline) / num_inline)
    inc_xline = int((max_xline - min_xline) / num_xline)
    inc_zsample = segy_file.bin[segyio.BinField.Interval] / 1000    
    
    shape = (trace_inlines_unique.size, trace_xlines_unique.size, num_zsamples)
    ti_idx = trace_inlines - trace_inlines.min()
    tx_idx = trace_xlines - trace_xlines.min()
    idx = np.arange(ti_idx.size)
    coords = np.dstack((ti_idx, tx_idx, idx))[0]
    coords = da.from_array(coords, chunks=(25, 3))
    
    with h5py.File(out_path, 'w') as f:
        
        dset = f.create_dataset(out_name, shape=shape)
        
        dset.attrs['dims'] = shape
        
        dset.attrs['inc_inline'] = inc_inline
        dset.attrs['inc_xline'] = inc_xline
        dset.attrs['inc_zsample'] = inc_zsample
        
        dset.attrs['min_inline'] = min_inline
        dset.attrs['min_xline'] = min_xline
        dset.attrs['min_zsample'] = min_zsample
        
        dset.attrs['max_inline'] = max_inline
        dset.attrs['max_xline'] = max_xline
        dset.attrs['max_zsample'] = max_zsample
                
        coords.map_blocks(write, segy_file, dset, dtype=np.float32).compute()
        


def segy_write(in_data, template_segy, out_file):
    
    cf(template_segy, out_file)
    
    with segyio.open(out_file, 'r+') as f:
        
        for i in range(in_data.shape[0]):
            try:
                il = f.ilines[i]
                f.iline[il] = in_data[i]
                
            except Exception:
                continue