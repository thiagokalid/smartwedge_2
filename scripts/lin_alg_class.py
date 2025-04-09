import numpy as np

#%% Question 2.1a

P = np.array([
    [1, 0],
    [2, -1],
    [0, 3]
])

Q = np.array([
    [1, 1, -1],
    [2, 0, 3]
])

R = np.array([
    [-1, 1, -4],
    [-4, 0, -6],
    [6, 6, -6]
])

# solve: PXQ = R
# (P.T @ P) @ X @ (Q @ Q.T) = P.T @ R @ Q.T
# X = (P.T @ P) ^-1 @ P.T @ R @ Q.T (Q @ Q.T)^-1

Xa = np.linalg.inv(P.T @ P) @ P.T @ R @ Q.T @ np.linalg.inv(Q @ Q.T)

print("(P @ X @ Q) - R = \n", (P @ Xa @ Q) - R)

#%% Question 2.1b

S = np.array([
    [1, 6],
    [3, 1]
])

Xb = lambda s,t : np.array([
    [13 - 6*s, -6 + 3*s, s],
    [5 - 6*t, -1 + 3*t, t]
])

print("(X@P) - S = \n", (Xb(0, 0) @ P) - S)
