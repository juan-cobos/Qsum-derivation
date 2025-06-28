import numpy as np

np.random.seed(43)

BETA = 5            #  β
num_states = 4      # |S|
num_actions = 3     # |A|

Q_i_joint = np.random.random((num_states, num_states, num_actions))  # |S| x |S| x |A|
Q_j_joint = np.random.random((num_states, num_states, num_actions))  # |S| x |S| x |A|

def softmax(x, axis):
    exps =  np.exp(BETA * x)
    print(exps.shape)
    return exps / np.sum(exps, axis=axis, keepdims=True)

# P_i(s, s', a)
P_i_joint = softmax(Q_i_joint, axis=2) # Softmax over action axis
# P_j(s, s', a)
P_j_joint = softmax(Q_j_joint, axis=2) # Softmax over action axis

# Uses np.allclose to avoid floating-point errors
assert np.allclose(np.sum(P_i_joint, axis=2), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_i_joint, axis=2)}."
assert np.allclose(np.sum(P_j_joint, axis=2), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_j_joint, axis=2)}."
print(P_i_joint.shape)
print(P_i_joint)
print(P_j_joint.shape)
print(P_j_joint)
P_i_joint = P_i_joint.reshape((num_states**2, num_actions)) # P_i(s*s', a)
P_j_joint = P_j_joint.reshape((num_states**2, num_actions)) # P_j(s*s', a)
print(P_i_joint.shape)
print(P_i_joint)
print(P_j_joint.shape)
print(P_j_joint)

P_joint = np.dot(P_i_joint, P_j_joint.T)
P_joint /= np.sum(P_joint, axis=1, keepdims=True)
assert np.allclose(np.sum(P_joint, axis=1), 1.0), \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(P_joint, axis=2)}."
print(P_joint)

# Compute steady-state vector via simulation: it results in uniform distribution (prob. due to uniform init)
steady_state = np.ones(num_states**2)/num_states**2 # Start with uniform distribution
for _ in range(10):
    steady_state = np.dot(steady_state, P_joint.T)
    #print(steady_state)
steady_state /= np.sum(steady_state, keepdims=True)
print(steady_state)
assert np.sum(steady_state) == 1.0, \
    f"Normalization error: the sum of probabilities must be 1, but got {np.sum(steady_state)}."
# Reshape steady_state to sum up over the other axis
steady_state = steady_state.reshape(num_states, num_states)
P_self = np.sum(steady_state, axis=1) # P(s)
print("P_self", P_self)
P_other = np.sum(steady_state, axis=0) # P(s')
print("P_other", P_other)

# Apply softmax inverse to P(s) and P(s')
def inverse_softmax(probs, beta=1.0, center=True):
    logits = (np.log(probs) - np.min(probs)) / beta   # Undo the softmax exponentiation and scaling
    return logits

Q_self = inverse_softmax(P_self, beta=BETA, center=False)
print(Q_self)
Q_other = inverse_softmax(P_other, beta=BETA, center=False)
print(Q_other)

# Recover Q_sum from Q_self, Q_other and W
w = 0.5
Q_sum = np.zeros((num_states, num_states, num_actions))
for i, si in enumerate(Q_self):
    for j, sj in enumerate(Q_other):
        Q_sum[i, j, :] = (1 - w) * si + w * sj
print(Q_sum)
