import numpy as np

np.random.seed(43)

BETA = 5            #  β
num_states = 3      # |S|
num_actions = 3     # |A|

Q_joint = np.random.random((num_states, num_states, num_actions))  # |S| x |S| x |A|

def softmax(x, axis):
    exps =  np.exp(BETA * x)
    return exps / np.sum(exps, axis=axis, keepdims=True)

# P(s, s', a)
P_joint = softmax(Q_joint, axis=2) # Softmax over action axis
# Uses np.allclose to avoid floating-point errors
assert np.allclose(np.sum(P_joint, axis=2), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_joint, axis=2)}."

P_joint_reshaped = P_joint.reshape(num_states**2, num_actions)
joint_trans_matrix = P_joint_reshaped @ P_joint_reshaped.T
joint_trans_matrix /= np.sum(joint_trans_matrix, axis=1)
print(joint_trans_matrix)

# Transpose to find right eigenvectors of P^T (which are left eigenvectors of P)
eigvals, eigvecs = np.linalg.eig(joint_trans_matrix.T)

# Find the eigenvector corresponding to eigenvalue 1
idx = np.argmin(np.abs(eigvals - 1.0))

# Extract and normalize the steady-state distribution
steady_state_dist = eigvecs[:, idx].real  # take real part in case of complex values
print("SS", steady_state_dist)

steady_state_dist /= np.sum(steady_state_dist)  # normalize to sum to 1
print("SS", steady_state_dist)

steady_state_dist_reshaped = steady_state_dist.reshape((num_states, num_states))
s_steady_state = np.sum(steady_state_dist_reshaped, axis=1)
s_prime_steady_state = np.sum(steady_state_dist_reshaped, axis=0)

print(s_steady_state)

# P(s, a)
P_self =  np.sum(P_joint, axis=1) # P_self = sum over s' (axis=1)
P_self /= np.sum(P_self, axis=1)  # Normalize to probs along axis 1
assert np.allclose(np.sum(P_self, axis=1), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_self, axis=1)}."

# P(s', a)
P_other =  np.sum(P_joint, axis=0) # P_self = sum over s (axis=0)
P_other /= np.sum(P_other, axis=1) # Normalize to probs along axis 1
assert np.allclose(np.sum(P_other, axis=1), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_other, axis=1)}."

print("P_joint \n", P_joint)
print("P_self \n", P_self)
print("P_other \n", P_other)




"""
P_self =  np.sum(P_joint, axis=1) # P_self = sum over s' (axis=1)
print(P_self)
print(P_self.shape)
P_other = np.sum(P_joint, axis=0) # P_other = sum over s axis
P_sum = np.sum(P_joint, axis=1)
"""
