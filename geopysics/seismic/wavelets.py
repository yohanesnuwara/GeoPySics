import numpy as np

def Ricker(f, t):
    assert len(f) == 1, 'Ricker wavelet needs 1 frequency as input'
    # f = f[0]
    pift = np.pi * f * t
    wav = (1 - 2 * pift ** 2) * np.exp(-pift ** 2)
    return wav

def Ormsby(f, t):
    assert len(f) == 4, 'Ormsby wavelet needs 4 frequencies as input'
    f = np.sort(f)  # Ormsby wavelet frequencies must be in increasing order
    pif = np.pi * f
    den1 = pif[3] - pif[2]
    den2 = pif[1] - pif[0]
    term1 = (pif[3] * np.sinc(pif[3] * t)) ** 2 - (pif[2] * np.sinc(pif[2])) ** 2
    term2 = (pif[1] * np.sinc(pif[1] * t)) ** 2 - (pif[0] * np.sinc(pif[0])) ** 2

    wav = term1 / den1 - term2 / den2
    return wav
  
def Ormsby(f, t):
  f1, f2, f3, f4 = f
  A = ((np.pi*f4)**2 / (np.pi*f4 - np.pi*f3)) * (np.sinc(f4*t)**2)
  B = ((np.pi*f3)**2 / (np.pi*f4 - np.pi*f3)) * (np.sinc(f3*t)**2)
  C = ((np.pi*f2)**2 / (np.pi*f2 - np.pi*f1)) * (np.sinc(f2*t)**2)
  D = ((np.pi*f1)**2 / (np.pi*f2 - np.pi*f1)) * (np.sinc(f1*t)**2)
  wav = (A-B)-(C-D)
  wav = wav / max(wav)
  return wav

def Klauder(f, t, T=5.0):
    assert len(f) == 2, 'Klauder wavelet needs 2 frequencies as input'

    k = np.diff(f) / T
    f0 = np.sum(f) / 2.0
    wav = np.real(np.sin(np.pi * k * t * (T - t)) / (np.pi * k * t) * np.exp(2 * np.pi * 1j * f0 * t))
    wav = wav / max(wav)    
    return wav
