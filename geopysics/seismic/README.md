# SeisTool

Python utilities for seismic data processing, attribute calculation, and inversion. 

* Reading 3D seismic data, static viewer, and interactive viewer with `seistool.openSegy3D`, `seistool.sliceCube`, and `seistool.sliceViewer`. 
* Seismic attribute calculation with `seistool.sliceAttribute`.

<!--

# Documentation for `seis_package`

`module`<br>
|_____ `function 1`<br>
|_____ `function 2`<br>
|_____ `function 3`<br>

[`seis_util`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md#seis_util)<br>
|_____ [`slice_cube`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md#slice_cube): Slicing a 3D seismic cube into inline, crossline, or timeslice 2D section. Then, display it.<br>
|_____ [`frequency_spectrum`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md#frequency_spectrum): Computing frequency spectrum of the whole 3D seismic cube, or 2D sections (inline or crossline)<br>

`seis_attribute`<br>
|_____ [`attribute_3d`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md#attribute_3d): Computing attribute of a 2D seismic data<br><br>
|_____ [`attribute_2d`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md#attribute_2d): Computing attribute of a 3D seismic cube<br>
|_____ `display_attribute`: Display into a computed attribute section

# [`segyio_read.py`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/segyio_read.py) program

```
Inline range from 100 to 750
Crossline range from 300 to 1250
TWT from 4.0 to 1848.0
Sample rate: 4.0 ms
'The 99th percentile is 6517; the max amplitude is 32767'
```

# `seis_util`

`from seis_util import *`

## `slice_cube`

### Structure of `slice_cube` function

```
slice_cube(cube, type='il', 
           inline_loc=400, inline_array=None,
           xline_loc=None, xline_array=None,
           timeslice_loc=None, timeslice_array=None,
           display='Yes', cmap='gray', figsize=None, vmin=None, vmax=None)
```

### Input variables

|**Variable**|**Type**|**Description**|**Options**|
|:--:|:--:|:--:|:--:|
|`cube`|3D numpy array|3D seismic data output of `segyio.tools.cube`||
|`type`|string|specify the type of slice|`il` for inline<br>`xl` for crossline<br>`ts` for timeslice|
|`inline_loc`|integer|preferred location of inline, if `type='il'` is chosen||
|`xline_loc`|integer|preferred location of crossline, if `type='xl'` is chosen||
|`timeslice_loc`|integer|preferred location of timeslice, if `type='ts'` is chosen||
|`inline_array`|1D numpy array|array of inline locations<br> output of `segyio.read`||
|`xline_array`|1D numpy array|array of crossline locations<br> output of `segyio.read`||
|`timeslice_array`|1D numpy array|array of timeslice locations<br> output of `segyio.read`||
|`display`|string|display option|<ul> <li>`Yes`display slice</li> <li>`No`don't display slice<br>output only `slice`</li> </ul>|
|`cmap`|string|(Only for `Yes` display) colormap option, default is `gray`|`seismic`, `RdBu`, `PuOr`, `Accent`, etc.|
|`figsize`|2-size tuple|(Only for `Yes` display) figure size, default is `None`|example: `(20,10)`|
|`vmin`|float/int|(Only for `Yes` display) lowest limit for colormap, default is `None`|example: 99th percentile value, see `segyio_read.py`|
|`vmax`|float/int|(Only for `Yes` display) upper limit for colormap, default is `None`|example: 99th percentile value, see `segyio_read.py`|

### Use

**Slicing at inline 400 of a 3D seismic cube WITHOUT any display of it (`display='No'`)**

> No need to specify `crossline_array` and `timeslice_array` for this option.

```
slice_cube(cube=data, type='il', inline_loc=400, inline_array=inlines, display='No')
```

**Slicing at inline 400 of a 3D seismic cube WITH the display of it (`display='Yes'`)**

> Must specify all `inline_array`, `crossline_array` and `timeslice_array` for this option.

Display specification:
* Colormap `seismic`
* Size figure `(20,10)`
* Lower and upper limit are the 99th percentiles of the 3D cube data, `vm`, as output of `segyio.read` [(See section above)]()

```
slice_cube(cube=data, type='il', inline_loc=400, inline_array=inlines, crossline_array=crosslines, timeslice_array=twt, display='Yes', cmap='gray', figsize=(20,10), vmin=-vm, vmax=vm)
```

### Outputs

The outputs of two options are different. The use of variable option `display='Yes'` results BOTH `slice` and the display of seismic in `plt.show`, while the use of variable option `display='No'` results ONLY the `slice`. 

**Slicing at inline 400 of a 3D seismic cube WITHOUT any display of it (`display='No'`)**
```
array([[    0.,     0.,     0., ...,   408.,  1717.,   389.],
       [    0.,     0.,     0., ...,   221.,  1021.,  1023.],
       [    0.,     0.,     0., ..., -1242.,   714.,  1032.],
       ...,
       [    0.,   -78.,  -101., ..., -1076.,   -84.,  1029.],
       [    0.,   -17.,   -35., ...,  2061.,   722., -1503.],
       [    0.,  -500.,   417., ...,  1331.,  1570.,  1016.]],
      dtype=float32)
```

**Slicing at inline 400 of a 3D seismic cube WITH the display of it (`display='Yes'`)**

![image](https://user-images.githubusercontent.com/51282928/83354775-1108f900-a385-11ea-9ba9-9c8866a62cbd.png)

## `frequency_spectrum`

### Structure of `frequency_spectrum`

```
frequency_spectrum(data, type='il', inline_array=None, xline_array=None, timeslice_array=None, sample_rate=0.004)
```

### Input Variables

|**Variable**|**Type**|**Description**|**Options**|
|:--:|:--:|:--:|:--:|
|`cube`|3D numpy array|3D seismic data output of `segyio.tools.cube`||
|`type`|string|specify the type of slice|`il` for inline<br>`xl` for crossline<br>`ts` for timeslice|
|`inline_loc`|integer|preferred location of inline, if `type='il'` is chosen||
|`xline_loc`|integer|preferred location of crossline, if `type='xl'` is chosen||
|`timeslice_loc`|integer|preferred location of timeslice, if `type='ts'` is chosen||
|`inline_array`|1D numpy array|array of inline locations<br> output of `segyio.read`||
|`xline_array`|1D numpy array|array of crossline locations<br> output of `segyio.read`||
|`timeslice_array`|1D numpy array|array of timeslice locations<br> output of `segyio.read`||
|`sample_rate`|float/int|sampling rate in milliseconds<br> output of `segyio.read`||

### Use 

**Frequency spectrum at Inline 400**

```
frequency_spectrum(il_slices, type='il', inline_array=inlines, xline_array=crosslines, timeslice_array=twt, sample_rate=sample_rate)
```

**Frequency spectrum at Crossline 1000**

```
frequency_spectrum(xl_slices, type='xl', inline_array=inlines, xline_array=crosslines, timeslice_array=twt, sample_rate=sample_rate)
```

**Frequency spectrum of the whole cube**

```
frequency_spectrum(data, type='whole', inline_array=inlines, xline_array=crosslines, timeslice_array=twt, sample_rate=sample_rate)
```

### Output

The output of this function is `freq_seis` (frequency) and `spec_seis` (amplitude) computed by FFT. To plot the spectra, use `plt.show`. This is the spectra of inline 400, crossline 1000, and the whole cube.

![image](https://user-images.githubusercontent.com/51282928/83357640-c2b12580-a397-11ea-9e02-e51e09ed2f51.png)

# `seis_attribute`

`from seis_attribute import *`

## `attribute_3d`

`attribute_3d` is a function to compute various seismic attributes from a given 3D seismic cube input. 

This function returns two different output options. This option is passed in the input variable `output`, the value is either `2d` or `3d`. `2d` is used if the result we want is a 2D attribute slice, whereas `3d` is used if the result we want is a 3D attribute cube. 

For `2d` output, we specify which slice type we want to generate by passing the input variable `type`. Values are `il` if we want inline, `xl` for crossline, and `ts` for timeslice. The location of each slice is passed in each input variable `inline_loc`, `xline_loc`, and `timeslice_loc`.

### Structure of `attribute_3d` function

```
attribute_3d( cube, output='2d', type='il', 
              inline_loc=400, inline_array=None,
              xline_loc=1000, xline_array=None,
              timeslice_loc=1404, timeslice_array=None,
              attribute_class='CompleTrace', 
              attribute_type='cosphase', kernel=None, sample_rate=4, 
              dip_factor=10, axis=-1)
```

### Input variables

|**Variable**|**Type**|**Description**|**Options**|
|:--:|:--:|:--:|:--:|
|`cube`|3D numpy array|3D seismic data output of `segyio.tools.cube`||
|`type`|string|specify the type of slice|`il` for inline<br>`xl` for crossline<br>`ts` for timeslice|
|`inline_loc`|integer|preferred location of inline, if `type='il'` is chosen||
|`xline_loc`|integer|preferred location of crossline, if `type='xl'` is chosen||
|`timeslice_loc`|integer|preferred location of timeslice, if `type='ts'` is chosen||
|`inline_array`|1D numpy array|array of inline locations<br> output of `segyio.read`||
|`xline_array`|1D numpy array|array of crossline locations<br> output of `segyio.read`||
|`timeslice_array`|1D numpy array|array of timeslice locations<br> output of `segyio.read`||
|`attribute_class`|string|class of attribute|See list below|
|`attribute_type`|string|type of attribute depends on its class|See list below|
|`**spec`|any|specified inputs depends on its attribute type|See list below|

### List of Attributes (sorted alphabetically) & its Details

|**Attribute Class**|**Attribute Type**|**Description**|**Input**|**Specified Input**|**Output**|
|:--:|:--:|:--:|:--:|:--:|:--:|
|`Amplitude`|`fder`|first derivative|`cube`|`axis=-1`|`slice` or `cube`|
|`Amplitude`|`gradmag`|gradient magnitude<br> using Gaussian operator|`cube`|`sigmas=(1,1,1)`|`slice` or `cube`|
|`Amplitude`|`reflin`|reflection intensity|`cube`|`kernel=(1,1,9)`|`slice` or `cube`|
|`Amplitude`|`rms`|root mean square|`cube`|`kernel=(1,1,9)`|`slice` or `cube`|
|`Amplitude`|`sder`|second derivative|`cube`|`axis=-1`|`slice` or `cube`|
|`CompleTrace`|`ampacc`|amplitude acceleration|`cube`||`slice` or `cube`|
|`CompleTrace`|`ampcontrast`|relative amplitude contrast|`cube`||`slice` or `cube`|
|`CompleTrace`|`apolar`|apparent polarity|`cube`||`slice` or `cube`|
|`CompleTrace`|`cosphase`|cosine of instantaneous<br> phase|`cube`||`slice` or `cube`|
|`CompleTrace`|`domfreq`|dominant frequency|`cube`|`sample_rate=4`|`slice` or `cube`|
|`CompleTrace`|`enve`|envelope|`cube`||`slice` or `cube`|
|`CompleTrace`|`freqcontrast`|frequency change|`cube`|`sample_rate=4`|`slice` or `cube`|
|`CompleTrace`|`inband`|instantaneous bandwidth|`cube`||`slice` or `cube`|
|`CompleTrace`|`infreq`|instantaneous frequency|`cube`|`sample_rate=4`|`slice` or `cube`|
|`CompleTrace`|`inphase`|instantaneous phase|`cube`||`slice` or `cube`|
|`CompleTrace`|`quality`|quality factor|`cube`|`sample_rate=4`|`slice` or `cube`|
|`CompleTrace`|`resamp`|response amplitude|`cube`||`slice` or `cube`|
|`CompleTrace`|`resfreq`|response frequency|`cube`|`sample_rate=4`|`slice` or `cube`|
|`CompleTrace`|`resphase`|response phase|`cube`||`slice` or `cube`|
|`CompleTrace`|`sweet`|sweetness|`cube`|`sample_rate=4`|`slice` or `cube`|
|`DipAzm`|`dipgrad` *)|gradient dips from<br> inline, crossline,<br> and z-gradients|`cube`|`dip_factor=10`,<br> `kernel=(3,3,3)`|`slice` or `cube` <ul> <li>`il_dip`inline dip</li> <li>`xl_dip`crossline dip</li> </ul>|
|`DipAzm`|`gst` *)|inner product of<br> gradient structure<br> tensors (GST)|`cube`|`kernel`|`slice` or `cube`<br>inner products of gradient<br>`gi2`, `gj2`, `gk2`,<br> `gigj`, `gigk`, `gjgk`|
|`DipAzm`|`gstdip2d` *)|2D gradient dips<br> from GST|`cube`|`dip_factor=10`,<br> `kernel=(3,3,3)`|`slice` or `cube` <ul> <li>`il_dip`inline dip</li> <li>`xl_dip`crossline dip</li> </ul>|
|`DipAzm`|`gstdip3d`|3D gradient dips<br> from GST|`cube`||`slice` or `cube`|
|`DipAzm`|`gstazm3d`|3D azimuth<br> from GST|`cube`|`dip_factor=10`,<br> `kernel=(3,3,3)`|`slice` or `cube`|
|`EdgeDetection`|`chaos`|multi-trace chaos|`cube`|`kernel=(3,3,9)`|`slice` or `cube`|
|`EdgeDetection`|`curv` *)|volume curvature<br> from 3D seismic dips|`cube`|`dip_factor=10`,<br> `kernel=(3,3,3)`|`slice` or `cube`<br>curvatures<br>`H`, `K`, `Kmax`,<br> `Kmin`, `KMPos`, `KMNeg`|
|`EdgeDetection`|`eigen`|eigen semblance|`cube`|`kernel=(3,3,9)`|`slice` or `cube`|
|`EdgeDetection`|`gstdisc`|discontinuity from<br> eigenvalues of GST|`cube`|`kernel=(3,3,9)`|`slice` or `cube`|
|`EdgeDetection`|`semblance`|semblance|`cube`|`kernel=(3,3,9)`|`slice` or `cube`|

> *) This attribute passes more than one outputs, so when using this attribute, pass different outputs. For instance, for `dipgrad`, use `il_dip, xl_dip = attribute_3d(...)` rather than `result = attribute_3d(...)`. More on this, see the [**Use**]() section.

### Use

#### Preprocessing your Data

There is a dimension specification for the `cube` input into `attribute_3d` function. The required dimension of the `cube` must be **`(il_size, xl_size, twt_size)`**, where `il_size`, `xl_size`, and `twt_size` are the sizes of the inlines, crosslines, and TWT.

To know the size of your inlines, crosslines, and TWT, you could pass `len(...)` to each inline array `inlines`, crossline array `crosslines`, and TWT array `twt`. You already have these arrays from program [segyio_read](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/segyio_read.py). Then, pass the following.

```
il_size = len(inlines)
xl_size = len(crosslines)
twt_size = len(twt)
```
In the Dutch F3 data for instance, `il_size=651`, `xl_size=951`, and `twt_size=462`. So, the dimension of cube is `(651, 951, 462)`.

In other cases for other data, you may encounter after `segyio_read` that the cube dimension is in different order, for instance `(il_size, xl_size, twt_size)`. So, you have to transpose your cube first!

Run `cube.transpose((1,0,2))` to make your `il_size` and `xl_size` to switch, so at the end you will have the right dimension `(il_size, xl_size, twt_size)`.

#### Computing attributes

In function `attribute_3d` to compute the attributes, the general input is `cube`, the 3D seismic cube data. If you want a 3D attribute cube, pass `output='3d'`, and if you want a 2D attribute slice, pass `output='2d'` in the input. 

For `2d` output, you need to specify the slice types `type` as `'il'`, `'xl'`, or `'ts'`, and respective locations `inline_loc`, `xline_loc`, `timeslice_loc`. You need to specify all `inline_array`, `xline_array`, and `timeslice_array`.

Also, you encounter specified input variables `**spec` in each attrubute listed above. These variables are **optional**, meaning that if you don't define these variables, the function will compute its **default** values. 

For instance, see attribute type `sweetness`. The specified input is `sample_rate`. You may calculate the sampling rate from [`segyio_read`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/segyio_read.py) method. If you do not specify it, the program will still run with its default value `0.004` in ms. 

**Example 1. Create a 2D Sweetness Timeslice at 1404 ms**  

```
result = attribute_3d(cube=data, output='2d', type='ts', inline_loc=1404, inline_array=inlines, 
                      attribute_class='CompleTrace', attribute_type='sweetness',
                      sample_rate=0.002)
```

**Example 2. Create a 3D  Crossline at 1000**  

```
result = attribute_3d(cube=data, output='3d', type='ts', inline_loc=1404, inline_array=inlines, 
                      attribute_class='CompleTrace', attribute_type='sweetness',
                      sample_rate=0.002)
```

### Outputs

|`type`|Dimension|Example: Dutch F3|
|:--:|:--:|:--:|
|`il`|`(1, twt_size, xl_size)`|`(1, 462, 951)`|
|`xl`|`(1, twt_size, il_size)`|`(1, 462, 651)`|
|`ts`|`(1, il_size, xl_size)`|`(1, 651, 951)`|
