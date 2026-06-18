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

for n in range (-3,4):
    for m in range (-3,4):
        G = n*b1 + m*b2
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
                  n1 = g[i][0]
                  n2 = g[i][1]
                  grid32[n1, n2] = wavefunctions[i][n]
             rho += 2*np.abs(np.fft.ifft2(grid32))**2 # spin gives 2

# exchange correlation potential

for i in range():
     