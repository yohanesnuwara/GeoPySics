# GeoPySics

GeoPySics is sandbox of Python implementations on geophysics

## Gravity 

Regional and residual anomaly separation from Complete Bouguer Anomaly (CBA) using 1D Fast Fourier Transform (FFT) and 2D Second Vertical Derivative (SVD)

<div>
<img src="https://user-images.githubusercontent.com/51282928/77532985-932dfa80-6ec8-11ea-9e04-b39ba5536487.PNG" width="800"/>
</div>

## Seismic

Development of [SeisTool](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/README.md), a Python utilities collection for seismic data processing, attributes calculation, and inversion

<div>
<img src="https://user-images.githubusercontent.com/51282928/115356419-d6e75680-a1e5-11eb-8b06-404c99494957.png" width="800"/>
</div>

Development of GeosoftML, which applies ML for seismic interpretation, seismic inversion, and geomodeling.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

<!--

<div>
<img src="https://user-images.githubusercontent.com/51282928/77532985-932dfa80-6ec8-11ea-9e04-b39ba5536487.PNG" width="800"/>
</div>

### [See here](https://github.com/yohanesnuwara/computational-geophysics/tree/master/gravity)

Data: [Roosevelt Hotsprings Geothermal USA from open-source USGS data](https://github.com/yohanesnuwara/computational-geophysics/blob/master/gravity/data/Gravity_UTM.txt)

<div>
<img src="https://user-images.githubusercontent.com/51282928/77532692-f5d2c680-6ec7-11ea-9db7-8c0d61e295b6.PNG" width="600"/>
</div>

### Regional and residual anomaly separation using Fourier Transform and Moving Average

#### FFT 1D slice into 2D map [See notebook](https://github.com/yohanesnuwara/computational-geophysics/blob/master/gravity/notebooks/fft_moving_average_2D_slice.ipynb)

<div>
<img src="https://user-images.githubusercontent.com/51282928/77532829-40544300-6ec8-11ea-9426-261bae10a1e2.PNG" width="700"/>
</div>

<div>
<img src="https://user-images.githubusercontent.com/51282928/77571109-621cec80-6eff-11ea-8a94-341d459008e1.PNG" width="700"/>
</div>

#### FFT 2D with Radially Averaged Power Spectrum [See notebook]()

### Regional and residual anomaly separation using Second Vertical Derivative

[See notebook](https://github.com/yohanesnuwara/computational-geophysics/blob/master/gravity/notebooks/second_vertical_derivative.ipynb)

<div>
<img src="https://user-images.githubusercontent.com/51282928/77532985-932dfa80-6ec8-11ea-9e04-b39ba5536487.PNG" width="800"/>
</div>

## Seismic

### [`seis_util`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/seis_util.py): Library for basic processing of seismic data. See [Notebook](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/seis_util.ipynb)

> Use library `segyio` for seismic io

* `slicing`: to slice inline, crossline, or timeslice from 3D seismic cube
* `display_slice`: to display the seismic slices
* `frequency_spectrum`: to compute frequency spectrum of an inline, a crossline, or the whole 3D cube

Notebooks containing seismic processing and seismic attribute analysis of Dutch F3 seismic cube from [`open-geoscience-repository`](https://github.com/yohanesnuwara/open-geoscience-repository)

![Netherlands-F3-Nuwara](https://user-images.githubusercontent.com/51282928/83105174-5eb30680-a0e4-11ea-8d17-f4a7ecdcaf0f.png)

### [`seis_widget`](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/seis_widget.py): Interactive slider to display slices of 3D seismic cube. See [Notebook](https://github.com/yohanesnuwara/computational-geophysics/blob/master/seismic/seis_widget.ipynb)

![image](https://user-images.githubusercontent.com/51282928/83326031-1be75f00-a29b-11ea-8e83-883f7a6bc819.png)

### [`seis_attribute`]()

## Magnetic
Still empty.

## Electrical
Still empty.

## Electromagnetic
Still empty.
