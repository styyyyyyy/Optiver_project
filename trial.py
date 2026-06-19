import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
T = 20                  # number of periods
C = 8                   # seat capacity
B = 6                   # scholarship budget units
w = 1                   # scholarship cost per funded admit

v = 2.0                 # value weight on quality
kappa = 3.0             # penalty per unfilled seat at terminal time
gamma = 0.0             # extra terminal value on cumulative quality (kept simple here)

alpha_N, beta_N, delta_N = -2.0, 4.0, -0.05
alpha_S, beta_S, delta_S = -1.0, 4.0, -0.05

q_grid = np.linspace(0, 1, 201)

# probability mass on q, here uniform
q_probs = np.ones_like(q_grid) / len(q_grid)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pN(q, t):
    return sigmoid(alpha_N + beta_N * q + delta_N * (T - t))

def pS(q, t):
    return sigmoid(alpha_S + beta_S * q + delta_S * (T - t))

def rN(q):
    return v * q

def rS(q):
    return v * q - w

# -----------------------------
# Value function:
# V[t, c, b] = expected value starting at time t with c seats and b budget
# t = 1,...,T+1
# -----------------------------
V = np.zeros((T + 2, C + 1, B + 1))

# terminal value
for c in range(C + 1):
    for b in range(B + 1):
        V[T + 1, c, b] = -kappa * c

# store optimal action by (t,c,b,q_index)
# 0=Reject, 1=No scholarship, 2=Scholarship
policy = np.zeros((T + 1, C + 1, B + 1, len(q_grid)), dtype=int)

# -----------------------------
# Backward induction
# -----------------------------
for t in range(T, 0, -1):
    for c in range(C + 1):
        for b in range(B + 1):
            total_val = 0.0

            for iq, q in enumerate(q_grid):
                # Reject
                WR = V[t + 1, c, b]

                # Admit without scholarship
                if c >= 1:
                    pn = pN(q, t)
                    WN = pn * (rN(q) + V[t + 1, c - 1, b]) + (1 - pn) * V[t + 1, c, b]
                else:
                    WN = -1e12

                # Admit with scholarship
                if c >= 1 and b >= w:
                    ps = pS(q, t)
                    WS = ps * (rS(q) + V[t + 1, c - 1, b - w]) + (1 - ps) * V[t + 1, c, b]
                else:
                    WS = -1e12

                vals = np.array([WR, WN, WS])
                a_star = int(np.argmax(vals))

                policy[t, c, b, iq] = a_star
                total_val += q_probs[iq] * vals[a_star]

            V[t, c, b] = total_val

# -----------------------------
# Extract thresholds for one fixed state (c_fixed, b_fixed)
# -----------------------------
c_fixed = 5
b_fixed = 3

tau_N = []
tau_S = []

for t in range(1, T + 1):
    actions = policy[t, c_fixed, b_fixed, :]  # over q-grid

    # tau_N: first q where action is N or S
    idx_N = np.where(actions >= 1)[0]
    if len(idx_N) > 0:
        tau_N.append(q_grid[idx_N[0]])
    else:
        tau_N.append(np.nan)

    # tau_S: first q where action is S
    idx_S = np.where(actions == 2)[0]
    if len(idx_S) > 0:
        tau_S.append(q_grid[idx_S[0]])
    else:
        tau_S.append(np.nan)

# -----------------------------
# Plot the two threshold lines
# -----------------------------
time = np.arange(1, T + 1)

plt.figure(figsize=(8, 5))
plt.plot(time, tau_N, label='Admit threshold (no scholarship or scholarship)')
plt.plot(time, tau_S, label='Scholarship threshold')
plt.xlabel('Time')
plt.ylabel('Applicant quality threshold')
plt.title(f'Thresholds over time at fixed state c={c_fixed}, b={b_fixed}')
plt.legend()
plt.grid(True)
plt.show()