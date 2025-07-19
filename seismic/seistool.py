import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import segyio
import pylops

def openSegy3D(filename, iline=189, xline=193):
  """
  Open 3D seismic volume in SEGY or SGY format 

  NOTE: Default byte location iline=189, xline=193
        If it returns "openSegy3D cannot read the data", change iline and xline
        Usually, specifying iline=5, xline=21 works
  """
  try:
    with segyio.open(filename) as f:

      data = segyio.tools.cube(f)

      inlines = f.ilines
      crosslines = f.xlines
      twt = f.samples
      sample_rate = segyio.tools.dt(f) / 1000

      print('Successfully read \n')
      print('Inline range from', inlines[0], 'to', inlines[-1])
      print('Crossline range from', crosslines[0], 'to', crosslines[-1])
      print('TWT from', twt[0], 'to', twt[-1])   
      print('Sample rate:', sample_rate, 'ms')  

      try:
        rot, cdpx, cdpy = segyio.tools.rotation(f, line="fast")
        print('Survey rotation: {:.2f} deg'.format(rot))      
      except:
        print("Survey rotation not recognized")  

      cube = dict({"data": data,
                   "inlines": inlines,
                   "crosslines": crosslines,
                   "twt": twt,
                   "sample_rate": sample_rate})
      
      # See Stackoverflow: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
      class AttrDict(dict):
          def __init__(self, *args, **kwargs):
              super(AttrDict, self).__init__(*args, **kwargs)
              self.__dict__ = self  

      cube = AttrDict(cube)

  except:
    try:
      # Specify iline and xline to segyio where to read
      with segyio.open(filename, iline=iline, xline=xline) as f:

        data = segyio.tools.cube(f)

        inlines = f.ilines
        crosslines = f.xlines
        twt = f.samples
        sample_rate = segyio.tools.dt(f) / 1000

        print('Successfully read \n')
        print('Inline range from', inlines[0], 'to', inlines[-1])
        print('Crossline range from', crosslines[0], 'to', crosslines[-1])
        print('TWT from', twt[0], 'to', twt[-1])   
        print('Sample rate:', sample_rate, 'ms')  

        try:
          rot, cdpx, cdpy = segyio.tools.rotation(f, line="fast")
          print('Survey rotation: {:.2f} deg'.format(rot))      
        except:
          print("Survey rotation not recognized")  

        cube = dict({"data": data,
                     "inlines": inlines,
                     "crosslines": crosslines,
                     "twt": twt,
                     "sample_rate": sample_rate})
        
        # See Stackoverflow: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self  

        cube = AttrDict(cube)
    
    except:
      print("openSegy3D cannot read the data")
      cube = None

  return cube 

def parseHeader(filename):
  """
  Parse header of a SEGY seismic volume
  """
  import segyio
  import re
  import pandas as pd

  def parse_trace_headers(segyfile, n_traces):
      '''
      Parse the segy file trace headers into a pandas dataframe.
      Column names are defined from segyio internal tracefield
      One row per trace
      '''
      # Get all header keys
      headers = segyio.tracefield.keys
      # Initialize dataframe with trace id as index and headers as columns
      df = pd.DataFrame(index=range(1, n_traces + 1),
                        columns=headers.keys())
      # Fill dataframe with all header values
      for k, v in headers.items():
          df[k] = segyfile.attributes(v)[:]
      return df


  def parse_text_header(segyfile):
      '''
      Format segy text header into a readable, clean dict
      '''
      raw_header = segyio.tools.wrap(segyfile.text[0])
      # Cut on C*int pattern
      cut_header = re.split(r'C ', raw_header)[1::]
      # Remove end of line return
      text_header = [x.replace('\n', ' ') for x in cut_header]
      text_header[-1] = text_header[-1][:-2]
      # Format in dict
      clean_header = {}
      i = 1
      for item in text_header:
          key = "C" + str(i).rjust(2, '0')
          i += 1
          clean_header[key] = item
      return clean_header  

  
  with segyio.open(filename, ignore_geometry=True) as f:  
    # Load headers
    n_traces = f.tracecount    
    bin_headers = f.bin
    # text_headers = parse_text_header(f)
    trace_headers = parse_trace_headers(f, n_traces)

  return trace_headers 

# Function to get byte number of a given trace key
def get_byte(key_name):
  header = segyio.tracefield.keys
  return header[key_name]

def cube_constructor(data, inlines, crosslines, twt):
  """
  Convert seismic cube data to cube object

  INPUT:

  data: Cube data (3D array)
  inlines: List of inline numbers
  crosslines: List of crossline numbers
  twt: List of TWT
  NOTE: size of inlines, crosslines, twt must match shape of data

  OUTPUT:

  cube: Python object with attributes .data, .inlines, .crosslines, .twt
  """
  cube = dict({"data": data,
                "inlines": inlines,
                "crosslines": crosslines,
                "twt": twt})
  class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  
  cube = AttrDict(cube)
  return cube

def sliceCube(cube, type='il', 
              inline_loc=400, 
              xline_loc=None, 
              timeslice_loc=None, 
              display=False, 
              cmap='gray', figsize=None, vmin=None, vmax=None):
  
  """
  Slicing 3D seismic cube into 2D slice, and display it (optional)
  
  NOTE: Static display. If you want interactive display, use: sliceViewer

  INPUT:

  cube: 3D seismic output of openSegy3D

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

  display: Display option. Default is False
    * False : Your output is the slice (2D numpy array).
    * True  : Your output is plot of the slice.

  If your display=True, configure the display details:
    * cmap: Matplotlib colormaps. 
    * figsize: figure size (2-size tuple)
    * vmin: lowest limit for colormap 
    * vmax: upper limit for colormap 

  OUTPUT:

  slice : 2D numpy array,  if display=True is specified
  plot  : Seismic display, if display=False is specified
  """

  import numpy as np
  import matplotlib.pyplot as plt

  # Unwrap cube
  inline_array, xline_array, timeslice_array = cube.inlines, cube.crosslines, cube.twt
  cube = cube.data

  if type == 'il':
    id = np.where(inline_array == inline_loc)[0][0]
    inline_slice = cube[id,:,:]
    
    if display == False:
      return(inline_slice)
        
    if display == True:
      plt.figure(figsize=figsize)
      plt.title('Inline {}'.format(inline_loc), size=20, pad=20)
      extent = [xline_array[0], xline_array[-1], timeslice_array[-1], timeslice_array[0]]

      p1 = plt.imshow(inline_slice.T, cmap=cmap, aspect='auto', extent=extent,
                      vmin=vmin, vmax=vmax)

      plt.colorbar()
      plt.xlabel('Crossline', size=15); plt.ylabel('TWT', size=15)
#       plt.show()

  if type == 'xl':

    id = np.where(xline_array == xline_loc)[0][0]
    xline_slice = cube[:,id,:]

    if display == False:
      return(xline_slice)
    
    if display == True:
      plt.figure(figsize=figsize)
      plt.title('Crossline {}'.format(xline_loc), size=20, pad=20)
      extent = [inline_array[0], inline_array[-1], timeslice_array[-1], timeslice_array[0]]

      p1 = plt.imshow(xline_slice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax)

      plt.colorbar()
      plt.xlabel('Inline', size=15); plt.ylabel('TWT', size=15)
#       plt.show()
  
  if type == 'ts':

    id = np.where(timeslice_array == timeslice_loc)[0][0]
    tslice = cube[:,:,id]

    if display == False:
      return(tslice)

    if display == True:

      plt.figure(figsize=figsize)
      plt.title('Timeslice {} ms'.format(timeslice_loc), size=20, pad=20)
      extent = [inline_array[0], inline_array[-1], xline_array[-1], xline_array[0]]

      p1 = plt.imshow(tslice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax)

      plt.colorbar()
      plt.xlabel('Inline', size=15); plt.ylabel('Crossline', size=15)
      plt.xlim(min(inline_array), max(inline_array))
      plt.ylim(min(xline_array), max(xline_array))
      plt.axis('equal')
#       plt.show()  

def sliceViewer(cube, cube_name=" "):
  """
  Interactive viewer of 2D slice from a 3D seismic volume data

  NOTE: Interactive display. If you want static display, use: sliceCube

  """
  from ipywidgets import interact, interactive, fixed, interact_manual, ToggleButtons
  import ipywidgets as widgets
  import numpy as np
  import matplotlib.pyplot as plt

  # Access data and properties of cube
  data, inlines, crosslines, twt, sample_rate = cube.data, cube.inlines,\
                                                cube.crosslines, cube.twt,\
                                                cube.sample_rate                                         

  # Interactive plotting
  type = ToggleButtons(description='Selection',
                       options=['Inline','Crossline','Timeslice'])
  cmap_button = ToggleButtons(description='Colormaps',
                              options=['gray','seismic','RdBu','PuOr','Accent'])
  
  inline_loc = widgets.IntSlider(value=min(inlines), min=min(inlines), 
                                 max=max(inlines))
  xline_loc = widgets.IntSlider(value=min(crosslines), min=min(crosslines), 
                                max=max(crosslines))
  timeslice_loc = widgets.FloatSlider(value=min(twt), min=min(twt), 
                                      max=max(twt), step=sample_rate)
  vmin = widgets.FloatSlider(value=np.amin(data), min=np.amin(data), 
                             max=np.amax(data))
  vmax = widgets.FloatSlider(value=np.amax(data), min=np.amin(data), 
                             max=np.amax(data))

  @interact   
  def f(type=type, inline_loc=inline_loc, xline_loc=xline_loc,
        timeslice_loc=timeslice_loc, vmin=vmin, vmax=vmax, cmap=cmap_button):  
    
    if type == 'Inline':

      # with segyio.open(filename) as f: 
      inline_slice = data[inline_loc-inlines[0],:,:]  

      plt.figure(figsize=(20, 10))
      plt.title('{} Seismic at Inline {}'.format(cube_name, inline_loc))
      extent = [crosslines[0], crosslines[-1], twt[-1], twt[0]]

      p1 = plt.imshow(inline_slice.T, cmap=cmap, aspect='auto', extent=extent,
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()

      plt.xlabel('Crossline'); plt.ylabel('TWT')
      plt.show()

    if type == 'Crossline':

      # with segyio.open(filename) as f:
      xline_slice = data[:,xline_loc-crosslines[0],:]  

      plt.figure(figsize=(20, 10))
      plt.title('{} Seismic at Crossline {}'.format(cube_name, xline_loc))
      extent = [inlines[0], inlines[-1], twt[-1], twt[0]]

      p1 = plt.imshow(xline_slice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()

      plt.xlabel('Inline'); plt.ylabel('TWT')
      plt.show()
    
    if type == 'Timeslice':

      id = np.where(twt == timeslice_loc)[0][0]
      tslice = data[:,:,id]

      plt.figure(figsize=(8,10))
      plt.title('{} Seismic at Timeslice {} ms'.format(cube_name, timeslice_loc))
      extent = [inlines[0], inlines[-1], crosslines[-1], crosslines[0]]

      p1 = plt.imshow(tslice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()


      plt.xlabel('Inline'); plt.ylabel('Crossline')
      plt.gca().xaxis.set_ticks_position('top') # axis on top
      plt.gca().xaxis.set_label_position('top') # label on top
      plt.xlim(min(inlines), max(inlines))
      plt.ylim(min(crosslines), max(crosslines))
      plt.axis('equal')
      plt.show()   

def sliceAttribute(cube, output='2d', type='il', 
                   inline_loc=400, 
                   xline_loc=1000, 
                   timeslice_loc=1404, 
                   attribute_class='CompleTrace', 
                   attribute_type='cosphase',
                   kernel=None, sample_rate=4, dip_factor=10, axis=-1,                   
                   display=False, 
                   figsize=(10,5), cmap='plasma', vmin=None, vmax=None):
  
  """
  Computing attribute of a 3D seismic cube, output a 2D attribute slice 

  INPUT:

  cube: 3D seismic data output of segyio.tools.cube (3D numpy array)

  type: specify the type of slice
    * 'il' for inline
    * 'xl' for crossline
    * 'ts' for timeslice

  inline_loc: preferred location of inline, if you specify type='il', 
              no need to input xline_loc, timeslice_loc

  xline_loc: preferred location of crossline, if you specify type='xl', 
             no need to input inline_loc, timeslice_loc

  timeslice_loc: preferred location of timeslice, if you specify type='ts', 
                 no need to input inline_loc, xline_loc
  
  attribute_class: specify the class of attribute (string)
    * 'Amplitude': amplitude attributes
    * 'CompleTrace': complex trace attributes
    * 'DipAzm': dip and azimuth attributes
    * 'EdgeDetection': edge detection attributes

  attribute_type: specify the attribute type (string). 

  display: Option to display.
    * Default is False: No display, but outputs the calculated attribute in 2D/3D array
    * If True: Display the calculated attribute
  
  Each class has different attribute types, as follows.
    * For class 'Amplitude', attributes are:
      * 'fder': first derivative
      * 'sder': second derivative
      * 'rms': rms of trace value (e.g. amplitude)
      * 'gradmag': gradient magnitude using Gaussian operator
      * 'reflin': reflection intensity
    * For class 'CompleTrace', attributes are: 
      * 'enve': envelope
      * 'inphase': instantaneous phase 
      * 'cosphase': cosine instantaneous phase 
      * 'ampcontrast': relative amplitude contrast
      * 'ampacc': amplitude acceleration
      * 'infreq': instantaneous frequency 
      * 'inband': instantaneous bandwidth
      * 'domfreq': dominant frequency 
      * 'freqcontrast': frequency change 
      * 'sweet': sweetness 
      * 'quality': quality factor 
      * 'resphase': response phase 
      * 'resfreq': response frequency 
      * 'resamp': response amplitude 
      * 'apolar': apparent polarity
    * For class 'DipAzm', attributes are: 
      * 'dipgrad': gradient dips from inline, crossline, and z-gradients 
      * 'gst': the gradient structure tensors (GST), inner product of 
        gradients.
      * 'gstdip2d': 2D gradient dips from GST
      * 'gstdip3d': 3D gradient dips from GST
      * 'gstazm3d': 3D azimuth from GST
    
    * For class 'EdgeDetection', attributes are:
      * 'semblance': semblance
      * 'gstdisc': discontinuity from eigenvalues of GST
      * 'eigen': multi-trace semblance from 3D seismic incorporating the 
                analytic trace
      * 'chaos': multi-trace chaos from 3D seismic
      * 'curv': volume curvature from 3D seismic dips
  """

  import numpy as np
  import matplotlib.pyplot as plt
  import segyio

  from seistool import sliceCube

  from CompleTrace import ComplexAttributes
  from DipAzm import DipAzm
  from EdgeDetection import EdgeDetection
  from Frequency import Frequency
  from SignalProcess import SignalProcess

  def make_dask(darray, output, attribute_class, attribute_type, kernel):

    if output == '2d':

      if attribute_class == 'Amplitude':
        x = SignalProcess()
        darray, chunks_init = SignalProcess.create_array(x, darray, kernel, 
                                                         preview=None)
        darray = darray.T

      if attribute_class == 'CompleTrace':  
        x = ComplexAttributes()
        darray, chunks_init = ComplexAttributes.create_array(x, darray, kernel, 
                                                             preview=None)
        darray = darray.T 

      if attribute_class == 'DipAzm':
        x = DipAzm()
        darray, chunks_init = DipAzm.create_array(x, darray, kernel=None, 
                                                  preview=None)
        darray = darray.T
      
      if attribute_class == 'EdgeDetection':  
        x = EdgeDetection()
        darray, chunks_init = EdgeDetection.create_array(x, darray, kernel, 
                                                         preview=None)
        darray = darray.T      

    if output == '3d':

      if attribute_class == 'Amplitude':
        x = SignalProcess()
        darray, chunks_init = SignalProcess.create_array(x, darray, kernel, 
                                                         preview=None)

      if attribute_class == 'CompleTrace':  
        x = ComplexAttributes()
        darray, chunks_init = ComplexAttributes.create_array(x, darray, kernel, 
                                                             preview=None)

      if attribute_class == 'DipAzm':
        x = DipAzm()
        darray, chunks_init = DipAzm.create_array(x, darray, kernel=None, 
                                                  preview=None)
      
      if attribute_class == 'EdgeDetection':  
        x = EdgeDetection()
        darray, chunks_init = EdgeDetection.create_array(x, darray, kernel, 
                                                         preview=None)

    return(x, darray)


  def compute(x, darray, attribute_class, attribute_type, kernel, 
              sample_rate, dip_factor, axis):

    if attribute_class == 'Amplitude':

      if attribute_type == 'fder':
        result = SignalProcess.first_derivative(x, darray, axis=-1, 
                                                preview=None)
        return(result)
      
      if attribute_type == 'sder':
        result = SignalProcess.second_derivative(x, darray, axis=-1, 
                                                 preview=None)
        return(result)

      if attribute_type == 'rms':
        result = SignalProcess.rms(x, darray, kernel=(1,1,9), 
                                   preview=None)
        return(result)

      if attribute_type == 'gradmag':
        result = SignalProcess.gradient_magnitude(x, darray, sigmas=(1,1,1), 
                                                  preview=None)
        return(result)

      if attribute_type == 'reflin':
        result = SignalProcess.reflection_intensity(x, darray, kernel=(1,1,9), 
                                                    preview=None)
        return(result)

    if attribute_class == 'CompleTrace':  

      if attribute_type == 'enve':
        result = ComplexAttributes.envelope(x, darray, preview=None)
        return(result)

      if attribute_type == 'inphase':
        result = ComplexAttributes.instantaneous_phase(x, darray, preview=None)   
        return(result)

      if attribute_type == 'cosphase':
        result = ComplexAttributes.cosine_instantaneous_phase(x, darray, 
                                                              preview=None)   
        return(result)

      if attribute_type == 'ampcontrast':
        result = ComplexAttributes.relative_amplitude_change(x, darray, 
                                                             preview=None)
        return(result)

      if attribute_type == 'ampacc':
        result = ComplexAttributes.amplitude_acceleration(x, darray, 
                                                          preview=None)
        return(result)

      if attribute_type == 'infreq':
        result = ComplexAttributes.instantaneous_frequency(x, darray, 
                                                           sample_rate=4, 
                                                           preview=None)
        return(result)

      if attribute_type == 'inband':
        result = ComplexAttributes.instantaneous_bandwidth(x, darray, 
                                                           preview=None)
        return(result)

      if attribute_type == 'domfreq':
        result = ComplexAttributes.dominant_frequency(x, darray, sample_rate=4, 
                                                      preview=None)
        return(result)

      if attribute_type == 'freqcontrast':
        result = ComplexAttributes.dominant_frequency(x, darray, sample_rate=4, 
                                                      preview=None)
        return(result)

      if attribute_type == 'sweet':
        result = ComplexAttributes.sweetness(x, darray, sample_rate=4, 
                                             preview=None)
        return(result)

      if attribute_type == 'quality':
        result = ComplexAttributes.quality_factor(x, darray, sample_rate=4, 
                                                  preview=None)
        return(result)

      if attribute_type == 'resphase':
        result = ComplexAttributes.response_phase(x, darray, preview=None)
        return(result)

      if attribute_type == 'resfreq':
        result = ComplexAttributes.response_frequency(x, darray, sample_rate=4, 
                                                      preview=None)
        return(result) 

      if attribute_type == 'resamp':
        result = ComplexAttributes.response_amplitude(x, darray, preview=None)
        return(result)

      if attribute_type == 'apolar':
        result = ComplexAttributes.apparent_polarity(x, darray, preview=None)
        return(result)

    if attribute_class == 'DipAzm':

      if attribute_type == 'dipgrad':
        result = DipAzm.gradient_dips(x, darray, dip_factor=10, kernel=(3,3,3), 
                                      preview=None)
        return(result) # result is il_dip, xl_dip

      if attribute_type == 'gst':
        result = DipAzm.gradient_structure_tensor(x, darray, kernel, 
                                                  preview=None)
        return(result) # result is gi2, gj2, gk2, gigj, gigk, gjgk
      
      if attribute_type == 'gstdip2d':
        result = DipAzm.gst_2D_dips(x, darray, dip_factor=10, kernel=(3,3,3), 
                                    preview=None)
        return(result) # result is il_dip, xl_dip

      if attribute_type == 'gstdip3d':
        result = DipAzm.gst_3D_dip(x, darray, dip_factor=10, kernel=(3,3,3), 
                                   preview=None)
        return(result)

      if attribute_type == 'gstazm3d':
        result = DipAzm.gst_3D_azm(x, darray, dip_factor=10, kernel=(3,3,3), 
                                   preview=None)
        return(result)


    if attribute_class == 'EdgeDetection':  

      if attribute_type == 'semblance':
        result = EdgeDetection.semblance(x, darray, kernel=(3,3,9), 
                                         preview=None)
        return(result)

      if attribute_type == 'gstdisc':
        result = EdgeDetection.gradient_structure_tensor(x, darray, 
                                                         kernel=(3,3,9), 
                                                         preview=None)
        return(result)

      if attribute_type == 'eigen':
        result = EdgeDetection.eig_complex(x, darray, kernel=(3,3,9), 
                                           preview=None)
        return(result)

      if attribute_type == 'chaos':
        result = EdgeDetection.chaos(x, darray, kernel=(3,3,9), 
                                     preview=None)
        return(result)

      if attribute_type == 'curv':
        # compute first inline and xline dips from gst
        x = DipAzm()
        darray_il, darray_xl = DipAzm.gradient_dips(x, darray, dip_factor=10, 
                                                    kernel=(3,3,3), 
                                                    preview=None)
        # compute curvature
        result = EdgeDetection.volume_curvature(x, darray_il, darray_xl, 
                                                dip_factor=10, 
                                                kernel=(3,3,3), 
                                                preview=None) 
        return(result) # result is H, K, Kmax, Kmin, KMPos, KMNeg
  
  """
  Main Program
  """

  if output == '3d':

    # 3D cube is directly as input to compute attribute function

    darray = cube.data
    x, darray = make_dask(darray, output, attribute_class, 
                          attribute_type, kernel)
    result = compute(x, darray, attribute_class, attribute_type, kernel, 
                     sample_rate, dip_factor, axis)   
    
    return result

  if output == '2d':

    # slicing the 3D cube based on inline, crossline, or timeslice selection
    # processing input to attribute computation
    # then compute attribute 

    # Unwrap inlines, xlines, twt
    inline_array, xline_array, timeslice_array = cube.inlines, cube.crosslines, cube.twt

    if type == 'il':
      slices = sliceCube(cube, type, inline_loc=inline_loc)

      darray = np.reshape(slices, slices.shape + (1,))

      x, darray = make_dask(darray, output, attribute_class, 
                            attribute_type, kernel)
      result = compute(x, darray, attribute_class, attribute_type, 
                       kernel, sample_rate, dip_factor, axis)
      
      if display==False:
        # Outputs 2D attribute
        return result.compute()

      if display==True:
        # Display the attribute
        b_line, c_line = xline_array, timeslice_array

        trans_attr = result.T
        reshape = trans_attr.reshape((trans_attr.shape[0], -1))

        extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]

        plt.figure(figsize=figsize)        
        p1 = plt.imshow(reshape.T, vmin=vmin, vmax=vmax, aspect='auto', 
                        extent=extent, cmap=cmap)
        plt.colorbar(p1)        

    if type == 'xl':
      slices = sliceCube(cube, type, xline_loc=xline_loc)      

      darray = np.reshape(slices, slices.shape + (1,))  

      x, darray = make_dask(darray, output, attribute_class, 
                            attribute_type, kernel)
      result = compute(x, darray, attribute_class, attribute_type, 
                       kernel, sample_rate, dip_factor, axis)

      if display==False:
        # Outputs 2D attribute
        return result.compute()
        
      if display==True:
        # Display the attribute
        b_line, c_line = inline_array, timeslice_array

        trans_attr = result.T
        reshape = trans_attr.reshape((trans_attr.shape[0], -1))

        extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]

        plt.figure(figsize=figsize)        
        p1 = plt.imshow(reshape.T, vmin=vmin, vmax=vmax, aspect='auto', 
                        extent=extent, cmap=cmap)
        plt.colorbar(p1)          

    if type == 'ts':
      slices = sliceCube(cube, type, timeslice_loc=timeslice_loc)      

      darray = np.reshape(np.transpose(slices), 
                          (np.transpose(slices)).shape + (1,))

      x, darray = make_dask(darray, output, attribute_class, 
                            attribute_type, kernel)
      result = compute(x, darray, attribute_class, attribute_type, 
                       kernel, sample_rate, dip_factor, axis)

      if display==False:
        # Outputs 2D attribute
        return result.compute()
        
      if display==True:
        # Display the attribute
        b_line, c_line = inline_array, xline_array

        trans_attr = result.T
        reshape = trans_attr.reshape((trans_attr.shape[0], -1))

        extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
        
        plt.figure(figsize=figsize)
        p1 = plt.imshow(reshape, vmin=vmin, vmax=vmax, aspect='auto', 
                        extent=extent, cmap=cmap)
        plt.colorbar(p1)             

def plot2D(computed_attribute, cube, type, cmap='plasma', vmin=None, vmax=None):
  """
  Display 2D Results (from attribute or inversion)

  INPUT:

  computed_attribute: output from computation (3D Dask array)
  
  type: 'il' for inline, 'xl' for crossline, 'ts' for timeslice

  cmap: matplotlib pyplot colormaps ('gray', 'RdBu', 'seismic', 
        jet, Accent, ...)
        
  vmin, vmax: the minimum and maximum range for colormap. Many options:
  * None, None: normal and default plotting
  * specified vmin, vmax (e.g. vmin = 0, vmax = 1000)
  * vmin = -percentile99, vmax = +percentile99, percentiles of the cube
  """

  import numpy as np
  import matplotlib.pyplot as plt
  
  # Unwrap cube
  inline_array, xline_array, twt_array = cube.inlines, cube.crosslines, cube.twt

  if type == 'il':
    b_line, c_line = xline_array, twt_array 
    trans_attr = computed_attribute.T
    reshape = trans_attr.reshape((trans_attr.shape[0], -1))

    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(reshape.T, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)
    
  if type == 'xl':
    b_line, c_line = inline_array, twt_array     
    trans_attr = computed_attribute.T
    reshape = trans_attr.reshape((trans_attr.shape[0], -1))

    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(reshape.T, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)    

  if type == 'ts':
    b_line, c_line = inline_array, xline_array   
    trans_attr = computed_attribute.T
    reshape = trans_attr.reshape((trans_attr.shape[0], -1))

    extent = [b_line[0], b_line[-1], c_line[-1], c_line[0]]
    p1 = plt.imshow(reshape, vmin=vmin, vmax=vmax, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(p1)            


def sliceFluidFactor(near_cube, far_cube, type='il', 
                     inline_loc=400, 
                     xline_loc=1000, 
                     timeslice_loc=1404, 
                     crossplot=False):
  """
  Calculate Fluid Factor using Approximation on Near-Far Stack

  INPUT:

  near_cube, far_cube: Near and far stack cubes

  type: specify the type of slice
    * 'il' for inline
    * 'xl' for crossline
    * 'ts' for timeslice

  inline_loc: preferred location of inline, if you specify type='il', 
              no need to input xline_loc, timeslice_loc

  xline_loc: preferred location of crossline, if you specify type='xl', 
             no need to input inline_loc, timeslice_loc

  timeslice_loc: preferred location of timeslice, if you specify type='ts', 
                 no need to input inline_loc, xline_loc  
  
  crossplot: Option to display result of (Far-Near) vs. Near crossplot with
             regressed line
    * Default is False: No crossplot is shown. Numerical results are shown.
    * Specify as True: Crossplot is shown.
  
  OUTPUT:

  By default, the following output is produced (crossplot=False):
    * FF: Calculated Fluid Factor slice
  
  If "crossplot=True", crossplot display is produced.
  """
  from sklearn.linear_model import LinearRegression
  import matplotlib.pyplot as plt
  import numpy as np

  # Slice cube
  if type=='il':
    sliceNear = sliceCube(near_cube, type, inline_loc=inline_loc)
    sliceFar = sliceCube(far_cube, type, inline_loc=inline_loc)
  if type=='xl':  
    sliceNear = sliceCube(near_cube, type, xline_loc=xline_loc)
    sliceFar = sliceCube(far_cube, type, xline_loc=xline_loc)    
  if type=='ts':
    sliceNear = sliceCube(near_cube, type, timeslice_loc=timeslice_loc)
    sliceFar = sliceCube(far_cube, type, timeslice_loc=timeslice_loc)

  # Calculate FF
  near_data, far_data = np.ndarray.flatten(sliceNear), np.ndarray.flatten(sliceFar)
  diff = far_data - near_data       

  lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
  lr.fit(near_data.reshape(-1,1), diff.reshape(-1,1))
  coef = lr.coef_[0]
  R2 = lr.score(near_data.reshape(-1,1), diff.reshape(-1,1))

  if crossplot==True:
    plt.figure(figsize=(7,7))

    # Crossplot (Far-Near) vs Near
    plt.scatter(near_data, diff, color='blue', alpha=0.5)  

    # Regressed line
    xmin, xmax = min(near_data), max(near_data)
    x = np.linspace(xmin, xmax, 10)
    y = coef * x
    plt.plot(x, y, color='red', label="coef: {:.3f} \n $R^2$: {:.3f}".format(coef[0], R2))

    plt.legend(fontsize=12)
    plt.xlabel("Near Amplitude", size=12)
    plt.ylabel("Far-Near Amplitude Difference", size=12)
    plt.title("Fluid Factor Crossplot", size=20, pad=10)
    plt.gca().set_aspect('equal') 

  if crossplot==False:
    diff = sliceFar - sliceNear
    FF = diff - coef[0] * sliceNear
    return FF    

def rotate(origin, point, angle):
  """
  Rotate Seismic Survey

  INPUT:

  origin: Origin point or axis of rotation. Usually (min(x), min(y)) of CDP
  point: x and y coordinates of CDP
  angle: Angle of rotation

  OUTPUT:

  xrot, yrot: Rotated x and y coordinates of CDP
  """
  ox, oy = origin
  px, py = point

  xrot = ox + np.cos(np.deg2rad(angle)) * (px - ox) - np.sin(np.deg2rad(angle)) * (py - oy)
  yrot = oy + np.sin(np.deg2rad(angle)) * (px - ox) + np.cos(np.deg2rad(angle)) * (py - oy)
  return xrot, yrot

def extract_geobody(cube, value, range_x, range_y, range_z, 
                    figsize=(15,10), elev=90, azim=-90):
  """
  Extract geobody from an attribute cube

  INPUT:

  cube: Attribute cube object (3D array)
  value: Threshold value of attribute
  range_x: Min and max of x coordinate (Tuple)
  range_y: Min and max of y coordinate (Tuple)
  range_z: Min and max of z coordinate or TWT (Tuple)
  elev, azim: Viewing elevation and azimuth

  OUTPUT: 

  Plot of extracted geobodies
  """
  cube[cube<value] = 0
  cube[cube>value] = 1
  cube = np.swapaxes(cube, 1, 0)
  nx, ny, nz = cube.shape

  x = np.linspace(range_x[0], range_x[1], nx+1)
  y = np.linspace(range_y[0], range_y[1], ny+1)
  z = np.linspace(range_z[0], range_z[1], nz+1)
  x, y, z = np.meshgrid(y, x, z)

  def make_ax(grid=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    ax.invert_zaxis()
    # ax.view_init(60,45)
    ax.view_init(elev, azim)
    return ax

  ax = make_ax(True)
  ax.voxels(x, y, z, cube, facecolor='lime', shade=False, edgecolors='k', linewidth=0.2)
  plt.show()

def frequencySpectrum(cube, mini_range, dt=0.005, nfft=2**11, display=False):
  """
  Frequency Spectrum of a Mini-cube using Fast Fourier Transform

  NOTE: This operation can possibly consume lots of RAMs.
  """
  # Unwrap cube
  inline_array, xline_array, timeslice_array = cube.inlines, cube.crosslines, cube.twt
  data = cube.data

  # Unwrap minicube ranges
  il0, il1 = mini_range["il"][0], mini_range["il"][1]
  xl0, xl1 = mini_range["xl"][0], mini_range["xl"][1]
  ts0, ts1 = mini_range["twt"][0], mini_range["twt"][1]

  # Find indexes
  il0_id, il1_id = il0-inline_array[0], il1-inline_array[0]
  xl0_id, xl1_id = xl0-xline_array[0], xl1-xline_array[0]
  ts0_id = np.where(timeslice_array == ts0)[0][0]
  ts1_id = np.where(timeslice_array == ts1)[0][0]

  # Make mini cube
  d = data[il0_id:il1_id+1, xl0_id:xl1_id+1, ts0_id:ts1_id+1]

  wav_est_fft = np.mean(np.abs(np.fft.fft(d, nfft, axis=-1)), axis=(0, 1))
  fwest = np.fft.fftfreq(nfft, d=dt)

  if display==False:
    return fwest, wav_est_fft

  if display==True:
    plt.plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
    plt.xlim(min(fwest[:nfft//2]), max(fwest[:nfft//2]))
    plt.ylim(ymin=0)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum', size=20, pad=10)

def statisticalWavelet(cube, mini_range, dt=0.005, nfft=2**11, nt_wav=41,
                       display=False):
  # Frequency spectrum estimation with FFT
  fwest, wav_est_fft = frequencySpectrum(cube, mini_range)

  # time axis for wavelet
  t_wav = np.arange(nt_wav) * dt
  t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

  # Estimate wavelet
  wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
  wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
  wav_est = wav_est / wav_est.max()
  wcenter = np.argmax(np.abs(wav_est))

  if display==False:
    return t_wav, wav_est

  if display==True:
    plt.plot(t_wav, wav_est, 'k')
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude')
    plt.title('Statistical Wavelet', size=20, pad=10)

def slicePostStackInversion(data, type, wav, m0,
                           explicit=False, simultaneous=False, epsI=None,
                           epsR=None, dottest=False, epsRL1=None):
  """
  Inversion of post-stack seismic data (2D slice) using PyLops engine

  INPUT:

  data: 2D slice of inline, crossline, or timeslice

  type: Specify 'il' (inline), 'xl' (crossline), or 'ts' (timeslice)

  wav: Wavelet (can be statistical wavelet)

  m0: Background or initial model. Must have same dimension with data.

  The next 6 variables are 'pylops.avo.poststack.PoststackInversion' variables:
    * explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preferred for small data)
    * simultaneous : :obj:`bool`, optional
        Simultaneously invert entire data (``True``) or invert
        trace-by-trace (``False``) when using ``explicit`` operator
        (note that the entire data is always inverted when working
        with linear operator)
    * epsI : :obj:`float`, optional
        Damping factor for Tikhonov regularization term
    * epsR : :obj:`float`, optional
        Damping factor for additional Laplacian regularization term
    * dottest : :obj:`bool`, optional
        Apply dot-test
    * epsRL1 : :obj:`float`, optional
        Damping factor for additional blockiness regularization term
  NOTE: For more details see the original PyLops docs.

  OUTPUT:

  m_relative: Inverted AI (2D numpy array)
  r_relative: Residual of inverted AI (2D numpy array)
  """
  if type=='il':
    a, b = data.shape
    data = data.reshape((1,a,b))
    m0 = m0.reshape((1,a,b))

  if type=='xl':
    a, b = data.shape
    data = data.reshape((a,1,b))
    m0 = m0.reshape((a,1,b))

  # if type=='ts':
  #   a, b = data.shape
  #   data = data.reshape((a,b,1))
  #   m0 = m0.reshape((a,b,1))

  # Swap time axis to first dimension
  data = np.swapaxes(data, -1, 0)
  m0 = np.swapaxes(m0, -1, 0)

  # Inversion
  m_relative, r_relative = \
      pylops.avo.poststack.PoststackInversion(data, wav,
                                              m0=m0, explicit=explicit,
                                              simultaneous=simultaneous,
                                              epsI=epsI, epsR=epsR,
                                              dottest=dottest, epsRL1=epsRL1)

  # Swap time axis back to last dimension
  data = np.swapaxes(data, 0, -1)
  m_relative = np.swapaxes(m_relative, 0, -1)
  r_relative = np.swapaxes(r_relative, 0, -1)

  # # Reshaping results
  # m_relative = m_relative.reshape((data.shape[1], data.shape[2]))
  # r_relative = r_relative.reshape((data.shape[1], data.shape[2]))

  # Once again swap axis to condition output so that it can be plotted
  if type=='il':
    m_relative = np.swapaxes(m_relative, 1, 2)
    r_relative = np.swapaxes(r_relative, 1, 2)
  if type=='xl':
    m_relative = np.swapaxes(m_relative, 0, 2)
    r_relative = np.swapaxes(r_relative, 0, 2)

  # if type=='ts':
    # m_relative = np.nan

  return m_relative, r_relative
