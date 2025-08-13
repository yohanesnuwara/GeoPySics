def AkiRichards(Vp1, Vp2, Vs1, Vs2, rho1, rho2, theta):
  delta_Vp = Vp2 - Vp1
  delta_Vs = Vs2 - Vs1
  delta_rho = rho2 - rho1
  Vp, Vs, rho = np.mean([Vp1, Vp2]), np.mean([Vs1, Vs2]), np.mean([rho1, rho2])
  gamma = (Vs / Vp)**2
  A = 0.5 * ((delta_Vp / Vp) + (delta_rho / rho))
  B = 0.5 * (delta_Vp / Vp) - (4 * gamma * (delta_Vs / Vs)) - \
  (2 * gamma * (delta_rho / rho))
  C = 0.5 * (delta_Vp / Vp)
  theta = np.deg2rad(theta)
  R = A + (B * (np.sin(theta))**2) + (C * (np.sin(theta))**2 * (np.tan(theta))**2)
  return R, A, B, C

def Ricker(f, t):
  pift = np.pi * f * t
  wav = (1 - 2 * pift ** 2) * np.exp(-pift ** 2)
  return wav

def modelAVO(Vp1, Vp2, Vs1, Vs2, rho1, rho2, wavelet,
             start_angle=0, end_angle=30, n_sample_angle=100, n_sample_trace=100,
             cmap='cubehelix', interpolation=None, loc='best'):

  theta = np.linspace(start_angle, end_angle, n_sample_angle)

  # Calculate reflectivity as function of theta
  R, A, B, C = AkiRichards(Vp1, Vp2, Vs1, Vs2, rho1, rho2, theta)

  def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

  # Create reflectivity trace
  x = np.zeros((n_sample_trace, n_sample_angle))
  # x[(n_sample_trace//2)-1,:] = normalize(R)
  x[(n_sample_trace//2)-1,:] = R

  # Convolve with wavelet to produce seismic amplitude trace
  for i in range(n_sample_angle):
    x[:,i] = np.convolve(x[:,i], w, 'same')

  # Plot CMP gather response
  plt.subplot(1,2,1)
  plt.imshow(x, aspect='auto', extent=(min(theta), max(theta), n_sample_trace, 0),
             cmap=cmap, interpolation=interpolation)
  plt.ylim((n_sample_trace//2)+10,(n_sample_trace//2)-10)
  plt.xlabel("Angle [°]", size=15)
  plt.ylabel("Samples", size=15)
  plt.title("CMP Gather Response", size=20, pad=10)
  plt.colorbar()

  plt.subplot(1,2,2)
  plt.plot(theta, R, color='black', label='A: {:.4f}\nB: {:.4f}\nC: {:.4f}'.format(A, B, C))
  plt.axhline(0, ls='--', color='black', lw=1)
  plt.xlim(min(theta), max(theta))
  # plt.ylim(-0.2,0.2)
  plt.xlabel("Angle [°]", size=15)
  plt.ylabel("Reflectivity", size=15)
  plt.title("Amplitude vs. Angle", size=20, pad=10)
  plt.legend(loc=loc)

def extract_geobody(cube, value, range_x, range_y, range_z,
                    figsize=(10,15), elev=90, azim=-90, title='Geobody'):
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
  data = cube.copy()
  cube[cube>value] = 1
  cube[cube!=1] = 0
  # cube = np.swapaxes(cube, 1, 0)
  nx, ny, nz = cube.shape
  cube = np.array(cube, dtype=bool)

  x = np.linspace(range_x[0], range_x[1], nx+1)
  y = np.linspace(range_y[0], range_y[1], ny+1)
  z = np.linspace(range_z[0], range_z[1], nz+1)
  x, y, z = np.meshgrid(y, x, z)

  colors = plt.cm.plasma(data)

  xw = np.full(50, 3066)
  yw = np.full(50, 9340)
  zw = np.linspace(t0, t1, 50)

  def make_ax(grid=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_title(title, pad=20)
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=20)
    ax.set_zlabel("TWT", labelpad=20)
    ax.grid(grid)
    ax.invert_zaxis()
    # ax.view_init(60,45)
    ax.view_init(elev, azim)
    return ax

  ax = make_ax(True)
  ax.voxels(x, y, z, cube, facecolor='lime', shade=False, edgecolors='k', linewidth=0.2)
  # ax.plot(yw, xw, zw, lw=10)
  plt.show()
