from mayavi import mlab
import numpy as np

# Create a sphere
r = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()

# data = np.genfromtxt('leb.txt')
# xx, yy, zz = np.hsplit(data, 3)
import pandas as pd
point_sgd = pd.read_csv('./psi_hat_sgd.csv', sep=",", header=None).values.tolist()[:40]
xx = [item[0] for item in point_sgd]
yy = [item[1] for item in point_sgd]
zz = [item[2] for item in point_sgd]
point_qng = pd.read_csv('./psi_hat_qng.csv', sep=",", header=None).values.tolist()[:10]
xxx = [item[0] for item in point_qng]
yyy = [item[1] for item in point_qng]
zzz = [item[2] for item in point_qng]
mlab.mesh(x , y , z, color=(0.0,0.5,0.5))
mlab.points3d(xx, yy, zz, scale_factor=0.05)
mlab.points3d(xxx, yyy, zzz, scale_factor=0.05, color=(0.2,0.1,0.5))
mlab.plot3d([1, 0], [0, 0], [0, 1], color=(0,0,0), tube_radius=1.)
mlab.show()