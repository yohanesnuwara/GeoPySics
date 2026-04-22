def slice_cube(cube, type='il', 
               inline_loc=400, inline_array=None,
               xline_loc=None, xline_array=None,
               timeslice_loc=None, timeslice_array=None,
               display='Yes', cmap='gray', figsize=None, vmin=None, vmax=None):
  
  """
  Slicing 3D seismic cube into 2D slice, and display it (optional)
  (Copyright, Y. Nuwara, ign.nuwara97@gmail.com)

  Input:

  cube: 3D seismic data output of segyio.tools.cube (3D numpy array)
  type: specify the type of slice
    * 'il' for inline
    * 'xl' for crossline
    * 'ts' for timeslice)

  inline_loc: preferred location of inline, if you specify type='il', 
              no need to input xline_loc, timeslice_loc
  xline_loc: preferred location of crossline, if you specify type='xl', 
             no need to input inline_loc, timeslice_loc
  timeslice_loc: preferred location of timeslice, if you specify type='ts', 
                 no need to input inline_loc, xline_loc
  
  inline_array: array of inline locations (output from segyio.read)
  xline_array: array of xline locations (output from segyio.read)
  timeslice_array: array of timeslice locations (output from segyio.read)

  display: Display option, specify 
    * 'Yes' if you want to display your slice.
    * 'No' if you don't want to display your slice, just to obtain the slice.

  If you specify 'Yes', configure the display details:
    * cmap: colormaps (Default is 'gray'. Other options 'seismic', 'RdBu', 'PuOr', etc)
    * figsize: figure size (2-size tuple. Default is None)
    * vmin: lowest limit for colormap (float/integer. Default is None)
    * vmax: upper limit for colormap (float/integer. Default is None)

  Output:

  slice: 2D numpy array, the slice, if display='No' is specified
  plt.show (seismic display), if display='Yes' is specified

  """

  import numpy as np
  import matplotlib.pyplot as plt

  if type == 'il':
    id = np.where(inline_array == inline_loc)[0][0]
    inline_slice = cube[id,:,:]
    
    if display == 'No':
      return(inline_slice)
        
    if display == 'Yes':
      plt.figure(figsize=figsize)
      plt.title('Dutch F3 Seismic at Inline {}'.format(inline_loc), size=20, pad=20)
      extent = [xline_array[0], xline_array[-1], timeslice_array[-1], timeslice_array[0]]

      p1 = plt.imshow(inline_slice.T, cmap=cmap, aspect='auto', extent=extent,
                      vmin=vmin, vmax=vmax)

      plt.xlabel('Crossline', size=15); plt.ylabel('TWT', size=15)
      plt.show()

  if type == 'xl':

    id = np.where(xline_array == xline_loc)[0][0]
    xline_slice = cube[:,id,:]

    if display == 'No':
      return(xline_slice)
    
    if display == 'Yes':
      plt.figure(figsize=figsize)
      plt.title('Dutch F3 Seismic at Crossline {}'.format(xline_loc), size=20, pad=20)
      extent = [inline_array[0], inline_array[-1], timeslice_array[-1], timeslice_array[0]]

      p1 = plt.imshow(xline_slice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax)

      plt.xlabel('Inline', size=15); plt.ylabel('TWT', size=15)
      plt.show()
  
  if type == 'ts':

    id = np.where(timeslice_array == timeslice_loc)[0][0]
    tslice = cube[:,:,id]

    if display == 'No':
      return(tslice)

    if display == 'Yes':

      plt.figure(figsize=figsize)
      plt.title('Dutch F3 Seismic at Timeslice {} ms'.format(timeslice_loc), size=20, pad=20)
      extent = [inline_array[0], inline_array[-1], xline_array[-1], xline_array[0]]

      p1 = plt.imshow(tslice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax)

      plt.xlabel('Inline', size=15); plt.ylabel('Crossline', size=15)
      plt.xlim(min(inline_array), max(inline_array))
      plt.ylim(min(xline_array), max(xline_array))
      plt.axis('equal')
      plt.show()  

def frequency_spectrum(data, type='il', inline_array=None, xline_array=None, timeslice_array=None, sample_rate=0.004):

  """
  Compute the frequency spectrum of the whole cube, inline, or crossline
  cube
  (Copyright, Y. Nuwara, ign.nuwara97@gmail.com)

  Input:

  data: if you choose 'whole', data must be in 3D numpy array, if you choose
        either 'il' or 'xl', data must be in 2D numpy array (you should make
        the slice first using 'slicing' function)
  type: 'whole' for the whole cube, 'il' for inline section, 'xl' for
        crossline section 

  inline_array (inline), xline_array (xline), timeslice_array (time) 
  sample_rate: sampling rate of seismic data (in second) 

  Output:

  freq_seis: frequency
  spec_seis: seismic amplitude
  
  """

  import numpy as np
  import matplotlib.pyplot as plt

  if type == 'il':

    # transpose the slice 
    transp_slice = np.transpose(data)

    # take the average of each individual line traces

    # time
    min_time = 0
    max_time = len(timeslice_array)

    # crosslines
    xmin = 0
    xmax = len(xline_array)

    trace = np.mean(transp_slice[min_time:max_time, xmin:xmax], axis=1)

    Fs_seis = 1 / sample_rate  # Seconds.
    n_seis = len(trace)
    k_seis = np.arange(n_seis)
    T_seis = n_seis / Fs_seis
    freq_seis = k_seis / T_seis
    freq_seis = freq_seis[range(n_seis//2)]  # One side frequency range.

    spec_seis = np.fft.fft(trace) / n_seis  # FFT computing and normalization.
    spec_seis = spec_seis[range(n_seis//2)]

    # This is to smooth the spectrum over a window of 10.
    roll_win = np.ones(10) / 10
    spec_seis = np.convolve(spec_seis, roll_win, mode='same')

  if type == 'xl':

    # transpose the slice 
    transp_slice = np.transpose(data)

    # take the average of each individual line traces

    # time
    min_time = 0
    max_time = len(timeslice_array)

    # crosslines
    xmin = 0
    xmax = len(inline_array)

    trace = np.mean(transp_slice[min_time:max_time, xmin:xmax], axis=1)

    Fs_seis = 1 / sample_rate  # Seconds.
    n_seis = len(trace)
    k_seis = np.arange(n_seis)
    T_seis = n_seis / Fs_seis
    freq_seis = k_seis / T_seis
    freq_seis = freq_seis[range(n_seis//2)]  # One side frequency range.

    spec_seis = np.fft.fft(trace) / n_seis  # FFT computing and normalization.
    spec_seis = spec_seis[range(n_seis//2)]

    # This is to smooth the spectrum over a window of 10.
    roll_win = np.ones(10) / 10
    spec_seis = np.convolve(spec_seis, roll_win, mode='same')
  
  if type == 'whole':

    # transpose the cube
    transp_cube = np.transpose(data)

    # time
    min_time = 0
    max_time = len(timeslice_array)

    # crosslines
    xmin = 0
    xmax = len(xline_array)

    # inlines
    ymin = 0
    ymax = len(inline_array)

    mean_xl_traces = [] # mean of crossline traces of each inline section

    for i in range(ymax):
      mean_xl = np.mean(transp_cube[min_time:max_time, xmin:xmax, i], axis=1)
      mean_xl_traces.append(mean_xl)

    transp_xl = np.transpose(mean_xl_traces)

    # take average of each individual mean values of xl in the inline section

    trace = np.mean(transp_xl[min_time:max_time, ymin:ymax], axis=1)

    Fs_seis = 1 / sample_rate  # Seconds.
    n_seis = len(trace)
    k_seis = np.arange(n_seis)
    T_seis = n_seis / Fs_seis
    freq_seis = k_seis / T_seis
    freq_seis = freq_seis[range(n_seis//2)]  # One side frequency range.

    spec_seis = np.fft.fft(trace) / n_seis  # FFT computing and normalization.
    spec_seis = spec_seis[range(n_seis//2)]

    # This is to smooth the spectrum over a window of 10.
    roll_win = np.ones(10) / 10
    spec_seis = np.convolve(spec_seis, roll_win, mode='same')

  return(freq_seis, spec_seis)      
      
def slicing(cube, type, loc, a_line):

  """
  Slicing a seismic cube to inline section, crossline section, or timeslice section

  Input:

  cube: 3d numpy array, data (output of segyio read)
  type: 'il' for inline slicing, 'xl' for xline slicing, 'ts' for time slicing
  loc: the location of inline, crossline, or timeslice depends on the type
       you're choosing (integer/float)
  a_line: array of inline, crossline, and timeslice locations, depends on the 
          type you're choosing (1D numpy array)
  """

  import numpy as np

  if type == 'il':
    id = np.where(a_line == loc)[0][0]
    slices = cube[id,:,:]
  
  if type == 'xl':
    id = np.where(a_line == loc)[0][0]
    slices = cube[:,id,:]
  
  if type == 'ts':
    id = np.where(a_line == loc)[0][0]
    slices = cube[:,:,id]
  
  return(slices)

def display_slice(slices, type, b_line, c_line, cmap, vmin, vmax):

  """
  Display the slice

  Input:

  slices: the slice data, 2D numpy array
  type: 'il' for inline slicing, 'xl' for xline slicing, 'ts' for time slicing
  b_line, c_line: array of inline, crossline, and timeslice, depends on the 
                  type you're choosing (1D numpy array)
  * for 'il': b_line = crossline array, c_line = twt array
  * for 'xl': b_line = inline array, c_line = twt array
  * for 'ts': b_line = inline array, c_line = crossline array

  cmap: matplotlib pyplot colormaps ('gray', 'RdBu', 'seismic', 
        jet, Accent, ...)
  vmin, vmax: the minimum and maximum range for colormap. Many options:
  * None, None: normal and default plotting
  * specified vmin, vmax (e.g. vmin = 0, vmax = 1000)
  * vmin = -percentile99, vmax = +percentile99, percentiles of the cube

  """

  import numpy as np
  import matplotlib.pyplot as plt

  if type == 'il' or type == 'xl':
    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(slices.T, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)
   
  if type == 'ts':
    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(slices.T, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)

def attribute_input(slices, type):
  """
  Preparing the input array for attribute processing, after that passed to d2geo

  Input:

  slices: inline, crossline, or time slices output of function `slicing`
          (2D numpy array)
  type: 'il' for inline, 'xl' for crossline, 'ts' for timeslice

  Output:

  darray: 3D numpy array, that will be passed to d2geo attributes 
  """

  import numpy as np

  if type == 'il':
    darray = np.reshape(slices, slices.shape + (1,))
  
  if type == 'xl':
    darray = np.reshape(slices, slices.shape + (1,))    
  
  if type == 'ts':
    darray = np.reshape(np.transpose(slices), (np.transpose(slices)).shape + (1,))
  
  return(darray)

def display_attribute(computed_attribute, type, b_line, c_line, cmap, vmin, vmax):
  """
  Processing the output from d2geo attribute for matplotlib display

  Input:

  computed_attribute: output from the attribute computation (3D Dask array)
  type: 'il' for inline, 'xl' for crossline, 'ts' for timeslice

  b_line, c_line: array of inline, crossline, and timeslice, depends on the 
                  type you're choosing (1D numpy array)
  * for 'il': b_line = crossline array, c_line = twt array
  * for 'xl': b_line = inline array, c_line = twt array
  * for 'ts': b_line = inline array, c_line = crossline array

  cmap: matplotlib pyplot colormaps ('gray', 'RdBu', 'seismic', 
        jet, Accent, ...)
  vmin, vmax: the minimum and maximum range for colormap. Many options:
  * None, None: normal and default plotting
  * specified vmin, vmax (e.g. vmin = 0, vmax = 1000)
  * vmin = -percentile99, vmax = +percentile99, percentiles of the cube

  Output:

  attribute_slice: 2D Numpy array
  """

  import numpy as np
  import matplotlib.pyplot as plt

  if type == 'il' or type == 'xl':
    trans_attr = computed_attribute.T
    reshape = trans_attr.reshape((trans_attr.shape[0], -1))

    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(reshape.T, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)

  if type == 'ts':
    trans_attr = computed_attribute.T
    reshape = trans_attr.reshape((trans_attr.shape[0], -1))

    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(reshape, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)    
