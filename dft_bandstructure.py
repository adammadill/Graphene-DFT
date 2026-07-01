import numpy as np

# grid

a = 2.46/0.529177 # lattice parameter is 2.46 Angstroms, converted to Bohr radii for Hartree later then eV
a1 = np.array([a*np.sqrt(3)/2, a*1/2]) # lattice vectors
a2 = np.array([a*np.sqrt(3)/2, -a*1/2])

c1 = np.array([0,0]) # carbon atom positions in unit cell
c2 = np.array([a/np.sqrt(3), 0])

area = (a1[0]*a2[1] - a1[1]*a2[0]) # area of the unit cell

b1 = 2*np.pi*np.array([a2[1], -a2[0]])/area # reciprocal lattice vectors
b2 = 2*np.pi*np.array([-a1[1], a1[0]])/area

def calculate_bandstructure(N):
    
    g = []
    G_indices = []
    for n in range (-N,N+1):
        for m in range (-N,N+1):
            G = n*b1 + m*b2
            G_indices.append([n,m])
            g.append(G)

    # making hamiltonian

    K_0 = np.zeros((len(g), len(g)), dtype=complex) # kinetic

    for i in range  (len(g)):
             G_squared = g[i][0]**2 + g[i][1]**2
             K_0[i][i] = G_squared*0.5

    P_matrix = np.zeros((len(g), len(g)), dtype=complex) # potential

    for i in range (len(g)):
        for j in range (len(g)):
            G_diff = g[i] - g[j]
            efactor = np.exp(-1j*np.dot(G_diff, c1)) + np.exp(-1j*np.dot(G_diff, c2))
            V_G = - np.exp(-np.linalg.norm(G_diff)**2/1.0) # gaussian pseudopotential as first approximation, number in denom can be changed
            P_matrix[i][j] = V_G*efactor

    ham = P_matrix + K_0 

    # dft

    # 4 valence electrons per carbon and 2 carbon atoms per unit cell, so 8 valence electrons per unit cell

    converged = False
    old_energy = float('inf') # set to infinity to make sure first iteration runs
    iteration = 0

    while not converged:
        iteration += 1
        energies, wavefunctions = np.linalg.eigh(ham)
        current_total_energy = 2*np.sum(energies[:4]) # sum of the lowest 4 energies (2 electrons per spin)
        if not np.isfinite(current_total_energy):
            raise ValueError(f"Non-finite total energy at iteration {iteration}: {current_total_energy}")
        energy_diff = np.abs(current_total_energy - old_energy)
        print(f"Iteration {iteration}: Energy diff: {energy_diff:.6f}", flush=True)
        if energy_diff < 1e-6:
            converged = True
        else:
            rho = np.zeros((64,64))
            old_energy = current_total_energy
            grid64 = np.zeros((64,64), dtype=complex) # 64 to avoid nyquist and 64 for fft
            for n in range(4):
                grid64.fill(0.0)
                for i in range(len(g)):
                    n1 = G_indices[i][0]
                    n2 = G_indices[i][1]
                    grid64[n1 % 64, n2 % 64] = wavefunctions[i][n]
                rho += 2*np.abs(np.fft.ifft2(grid64))**2 # spin gives 2
            charge = np.sum(rho) # make sure rho is 8
            rho = rho * (8/charge)
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
                        # print("ran")
                        G_squared = Gx**2 + Gy**2
                        if G_squared < 1e-12:
                            V_hartree_ft[i][j] = 0
                        else:
                            V_hartree_ft[i][j] = 4*np.pi*rho_ft[i][j]/G_squared # by Poisson's equation in reciprocal space
            
            V_hartree = np.fft.ifft2(V_hartree_ft).real

            V_grid = V_xc_grid + V_hartree # grid potential xc and hartree

            # convert V_grid to reciprocal space N by N grid

            V_grid_ft = np.fft.fft2(V_grid)/ (64 * 64) # normalize by grid size
            matrix_N =  np.zeros((len(g), len(g)), dtype=complex)

            for i in range(len(g)):
                    for j in range(len(g)):
                        n1i = G_indices[i][0]
                        n2i = G_indices[i][1]
                        n1j = G_indices[j][0]
                        n2j = G_indices[j][1]
                        deltan1 = n1i - n1j
                        deltan2 = n2i - n2j
                        matrix_N[i, j] = V_grid_ft[deltan1, deltan2]

            ham = P_matrix + matrix_N + K_0 # total hamiltonian, took out K
    return P_matrix, matrix_N, g

# bandstructure 

Gamma = np.array([0.0, 0.0]) # Gamma point in reciprocal space
M = 0.5*b1 # M point in reciprocal space
K = (b1 - b2)/3 # K point in reciprocal space

# gamma - K

gammaKx = np.linspace(Gamma[0], K[0], 50)
gammaKy = np.linspace(Gamma[1], K[1], 50)

# K - M

KMx = np.linspace(K[0], M[0], 50)
KMy = np.linspace(K[1], M[1], 50)

# M - Gamma

MGx = np.linspace(M[0], Gamma[0], 50)
MGy = np.linspace(M[1], Gamma[1], 50)

all_kx = np.concatenate((gammaKx, KMx, MGx))
all_ky = np.concatenate((gammaKy, KMy, MGy))

k_points = np.column_stack((all_kx, all_ky))

# new kinetic energy matrix for each k-point for N=N

N=8

E_k_list = []
P_matrix_N, matrix_N_N, g_N = calculate_bandstructure(N)

# there is an issue where the energies are ordered incorrectly

from scipy.optimize import linear_sum_assignment

prev_wavefunc = None

for k in k_points:
     K_new = np.zeros((len(g_N), len(g_N)), dtype=complex)
     H_k = np.zeros((len(g_N), len(g_N)), dtype=complex)
     for i in range(len(g_N)):
            G_squared = (g_N[i][0] + k[0])**2 + (g_N[i][1] + k[1])**2
            K_new[i][i] = G_squared*0.5
     H_k = K_new + P_matrix_N + matrix_N_N
     E_k, wavefunc_k = np.linalg.eigh(H_k)
     # Order the energies correctly:
     if prev_wavefunc is not None:
         overlaps = np.abs(np.dot(prev_wavefunc.conj().T, wavefunc_k)) # overlap with previous
         #best_matches = np.argmax(overlaps, axis=1) # which new column matches old band
         _, best_matches = linear_sum_assignment(-overlaps) # one to one matching
         E_k = E_k[best_matches] # reorder
         wavefunc_k = wavefunc_k[:, best_matches]
     prev_wavefunc = wavefunc_k
     #E_k = np.sort(E_k)
     E_k_list.append(E_k*27.2114) # convert from Hartree to eV

energies_array = np.array(E_k_list)

# plotting

import matplotlib.pyplot as plt

# band structure along high symmetry points for N=N

plt.figure(figsize=(8,6))
plt.plot(energies_array[:, :5], color='blue') 
plt.xticks([0, 49, 99, 149], ['Γ', 'K', 'M', 'Γ'])
plt.ylabel('Energy (eV)')
plt.title('Band Structure of Graphene')
plt.savefig('graphene_band_structure.png', dpi=300)
#plt.show()

# fermi surface for N=5

k_x, k_y = np.meshgrid(np.linspace(-2.5, 2.5, 50), np.linspace(-2.5, 2.5, 50)) # pi/a placeholder doesnt matter how big it is

kx_flat = k_x.flatten()
ky_flat = k_y.flatten()

energies_3d = []

for i in range(len(kx_flat)):
     kx=kx_flat[i]
     ky=ky_flat[i]
     K_new = np.zeros((len(g_N), len(g_N)), dtype=complex)

     for j in range(len(g_N)):
        G_squared = (g_N[j][0] + kx)**2 + (g_N[j][1] + ky)**2
        K_new[j][j] = G_squared*0.5

     H_k = K_new + P_matrix_N + matrix_N_N
     E_k, wavefunc_k = np.linalg.eigh(H_k)
     energies_3d.append(E_k*27.2114) # convert to eV

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

#compare aliasing effect with matrix size
#compare to real bandstructure of graphene from literature

