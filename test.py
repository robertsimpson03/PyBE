#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import gaussian as g
import time
Lx = 300
Ly = 300

Nx = 501
Ny = 501

sigma_x = 20
sigma_y = 20.00001

print(sigma_y - sigma_x)
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
XX, YY = np.meshgrid(x,y)

t_start = time.perf_counter()
Ex, Ey = g.get_field(XX, YY, sigma_x, sigma_y)
speed = time.perf_counter() - t_start

print(f'Time: {speed}')

fig, axs = plt.subplots(2)

axs[0].pcolormesh(XX, YY, Ex)
axs[1].pcolormesh(XX, YY, Ey)

plt.show()
