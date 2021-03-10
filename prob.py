
import numpy as np
import matplotlib.pyplot as plt
g = 0.8
N = 200
J = np.random.normal(0, g/np.sqrt(N),[N,N])

eig = np.linalg.eigvals(J)

print(eig[1])
print(eig.shape)


ax = plt.gca()

ax.scatter(eig.real,eig.imag)
plt.xlabel('Real component')
plt.ylabel('Imaginary component')
plt.tight_layout()
plt.axis('square')
plt.savefig('prob_f1.png')
plt.close('all')
