import numpy as np

# grid

a = 2.46 # lattice parameter
a1 = np.array([a*np.sqrt(3)/2, a*1/2]) # lattice vectors
a2 = np.array([a*np.sqrt(3)/2, -a*1/2])

c1 = np.array([0,0])
c2 = np.array([a/np.sqrt(3), 0])

area = (a1[0]*a2[1] - a1[1]*a2[0]) # area of the unit cell

b1 = 2*np.pi*np.array([a2[1], -a2[0]])/area # reciprocal lattice vectors
b2 = 2*np.pi*np.array([-a1[1], a1[0]])/area

g = []
G_indices = []

for n in range (-4,5):
    for m in range (-4,5):
        G = n*b1 + m*b2
        G_indices.append([n,m])
        g.append(G)

# making hamiltonian

K_matrix = np.zeros((len(g), len(g)), dtype=complex) # kinetic

for i in range  (len(g)):
        G_squared = g[i][0]**2 + g[i][1]**2
        K_matrix[i][i] = G_squared*0.5

P_matrix = np.zeros((len(g), len(g)), dtype=complex) # potential

for i in range (len(g)):
     for j in range (len(g)):
          G_diff = g[i] - g[j]
          efactor = np.exp(-1j*np.dot(G_diff, c1)) + np.exp(-1j*np.dot(G_diff, c2))
          V_G = - np.exp(-np.linalg.norm(G_diff)**2/0.1) # gaussian pseudopotential
          P_matrix[i][j] = V_G*efactor

ham = K_matrix + P_matrix # hamiltonian

# dft

# 4 valence electrons per carbon and 2 carbon atoms per unit cell, so 8 valence electrons per unit cell

converged = False
old_energy = float('inf')

while not converged:
    energies, wavefunctions = np.linalg.eigh(ham)
    current_total_energy = 2*np.sum(energies[:4]) # sum of the lowest 4 energies (2 electrons per spin)
    energy_diff = np.abs(current_total_energy - old_energy)
    if energy_diff < 1e-6:
        converged = True
    else:
        rho = np.zeros((32,32))
        old_energy = current_total_energy
        grid32 = np.zeros((32,32), dtype=complex) # 32 to avoid nyquist and 32 for fft
        for n in range(4):
             grid32.fill(0.0)
             for i in range(len(g)):
                  n1 = G_indices[i][0]
                  n2 = G_indices[i][1]
                  grid32[n1, n2] = wavefunctions[i][n]
             rho += 2*np.abs(np.fft.ifft2(grid32))**2 # spin gives 2
        # exchange correlation potential
        V_xc_grid = np.zeros((len(rho), len(rho)))
        for i in range(len(rho)):
            for j in range(len(rho)):
                V_xc_grid[i][j] = - (3/np.pi)**(1/3) * rho[i][j]**(1/3)

        # hartree potential

        rho_ft = np.fft.fft2(rho)
        V_hartree_ft = np.zeros((len(rho), len(rho)), dtype=complex)

        for i in range(len(rho)):
            for j in range(len(rho)):
                if i == 0 and j == 0:
                    V_hartree_ft[i][j] = 0
                else:
                    if i < len(rho)//2:
                        k1 = i
                    else:
                        k1 = i - len(rho)
                    if j < len(rho)//2:
                        k2 = j
                    else:
                        k2 = j - len(rho)
                    Gx = k1*b1[0] + k2*b2[0]
                    Gy = k1*b1[1] + k2*b2[1]
                    G_squared = Gx**2 + Gy**2
                    V_hartree_ft[i][j] = 4*np.pi*rho_ft[i][j]/G_squared
        
        V_hartree = np.fft.ifft2(V_hartree_ft).real

        V_grid = V_xc_grid + V_hartree # grid potential xc and hartree

        # convert V_grid to reciprocal space 49 by 49 grid

        V_grid_ft = np.fft.fft2(V_grid)/ (32 * 32)
        matrix_49 = np.zeros((len(g), len(g)), dtype=complex)

        for i in range(len(g)):
                for j in range(len(g)):
                    n1i = G_indices[i][0]
                    n2i = G_indices[i][1]
                    n1j = G_indices[j][0]
                    n2j = G_indices[j][1]
                    deltan1 = n1i - n1j
                    deltan2 = n2i - n2j
                    matrix_49[i, j] = V_grid_ft[deltan1, deltan2]

        ham = K_matrix + P_matrix + matrix_49 # total hamiltonian

# bandstructure 

Gamma = np.array([0.0, 0.0]) # Gamma point in reciprocal space
M = 0.5*b1 # M point in reciprocal space
K = (2*b1 + b2)/3 # K point in reciprocal space

# gamma - M

gammaMx = np.linspace(Gamma[0], M[0], 50)
gammaMy = np.linspace(Gamma[1], M[1], 50)

# M - K

MKx = np.linspace(M[0], K[0], 50)
MKy = np.linspace(M[1], K[1], 50)

# K - Gamma

KGx = np.linspace(K[0], Gamma[0], 50)
KGy = np.linspace(K[1], Gamma[1], 50)

all_kx = np.concatenate((gammaMx, MKx, KGx))
all_ky = np.concatenate((gammaMy, MKy, KGy))

k_points = np.column_stack((all_kx, all_ky))

# new kinetic energy matrix for each k-point

E_k_list = []

for k in k_points:
     K_new = np.zeros((len(g), len(g)), dtype=complex)
     H_k = np.zeros((len(g), len(g)), dtype=complex)
     for i in range(len(g)):
            G_squared = (g[i][0] + k[0])**2 + (g[i][1] + k[1])**2
            K_new[i][i] = G_squared*0.5
     H_k = K_new + P_matrix + matrix_49
     E_k, wavefunc_k = np.linalg.eigh(H_k)
     E_k_list.append(E_k)

energies_array = np.array(E_k_list)

# plotting
 
import matplotlib.pyplot as plt

# band structure along high symmetry points

plt.figure(figsize=(8,6))
plt.plot(energies_array[:, :6], color='blue')  # Plot the first band
plt.xticks([0, 49, 99, 149], ['Γ', 'M', 'K', 'Γ'])
plt.ylabel('Energy (eV?)')
plt.title('Band Structure of Graphene')
plt.show()

# fermi surface

k_x, k_y = np.meshgrid(np.linspace(-2.5, 2.5, 50), np.linspace(-2.5, 2.5, 50)) # pi/a placeholder

kx_flat = k_x.flatten()
ky_flat = k_y.flatten()

energies_3d = []

for i in range(len(kx_flat)):
     kx=kx_flat[i]
     ky=ky_flat[i]
     K_new = np.zeros((len(g), len(g)), dtype=complex)

     for j in range(len(g)):
        G_squared = (g[j][0] + kx)**2 + (g[j][1] + ky)**2
        K_new[j][j] = G_squared*0.5

     H_k = K_new + P_matrix + matrix_49
     E_k, wavefunc_k = np.linalg.eigh(H_k)
     energies_3d.append(E_k)

energies_3d_array = np.array(energies_3d)
band3_grid = energies_3d_array[:, 3].reshape(k_x.shape) # third band
band4_grid = energies_3d_array[:, 4].reshape(k_x.shape) # fourth band

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(k_x, k_y, band3_grid, cmap='viridis')

# ax.plot_surface(k_x, k_y, band4_grid, cmap='plasma')

# ax.set_xlabel('kx')
# ax.set_ylabel('ky')
# ax.set_zlabel('Energy (eV)')

# plt.show()

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Surface(x=k_x, y=k_y, z=band3_grid, colorscale='Viridis', showscale=False))

fig.add_trace(go.Surface(x=k_x, y=k_y, z=band4_grid, colorscale='Plasma', showscale=False))

fig.update_layout(
    title='3D Band Structure of Graphene',
    scene=dict(
        xaxis_title='kx',
        yaxis_title='ky',
        zaxis_title='Energy (eV)'
    ),
    width=800,
    height=800
)

# Save the 3D plot as an interactive webpage file
fig.write_html("graphene_3d_bands.html")

#compare aliasing effect



