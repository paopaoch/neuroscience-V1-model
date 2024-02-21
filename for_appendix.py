import numpy as np
from math import atan2

class V1Connection:
    def __init__(self, m=200, n=200, kappa=np.pi/4, alpha=0.9):
        self.m = m
        self.n = n
        self.kappa = kappa
        self.alpha = alpha
        self.pref = np.linspace(0, 2*np.pi, m, endpoint=False)
    
    @staticmethod
    def spectral_abisca(A):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Need a square matrix")
        return np.max(np.real(np.linalg.eigvals(A)))
    
    def scale_by_spectral_abisca(self, A, alpha=None):
        if alpha is None:
            alpha = self.alpha
        spec_ab = self.spectral_abisca(A)
        scale = spec_ab / alpha
        return A / scale
    
    def cric_gauss(self, x):
        return np.exp((np.cos(x) - 1) / (self.kappa))
    
    @staticmethod
    def pref_diff(pref_a, pref_b):
        return pref_b[None, :] - pref_a[:, None]    

class NoRecurrenceWeights(V1Connection):
    def __init__(self, m=200, kappa=np.pi/4, alpha=0.9):
        n = m
        super().__init__(m, n, kappa, alpha)

    def get_weights(self):
        return np.zeros(shape=(self.m, self.n))
    
class RandomSymmetricConnectivityWeights(V1Connection):
    def __init__(self, m=200, kappa=np.pi/4, alpha=0.9):
        n = m
        super().__init__(m, n, kappa, alpha)
    
    def get_weights(self):
        W_tilde = np.random.randn(self.m, self.n)
        W = W_tilde + W_tilde.T
        return self.scale_by_spectral_abisca(W)

class SymmetricRingStructureWeights(V1Connection):
    def __init__(self, m=200, kappa=np.pi/4, alpha=0.9):
        n = m
        super().__init__(m, n, kappa, alpha)

    def get_weights(self):
        pref_matrix = self.pref_diff(self.pref, self.pref)
        W = self.cric_gauss(pref_matrix)
        return self.scale_by_spectral_abisca(W)

class BalanceRingStructureWeights(V1Connection):
    def __init__(self, m=200, kappa=np.pi/4, alpha=0.9):
        n = 2 * m
        super().__init__(m, n, kappa, alpha)

    def get_sub_weights(self):
        pref_matrix = self.pref_diff(self.pref, self.pref)
        W = self.cric_gauss(pref_matrix)
        return self.scale_by_spectral_abisca(W)
    
    def get_weights(self):
        EE = self.get_sub_weights()
        EI = - self.get_sub_weights()
        IE = self.get_sub_weights()
        II = - self.get_sub_weights()
        return np.concatenate((np.concatenate((EE, EI), axis=1),
                                np.concatenate((IE, II), axis=1)), axis=0)

class NetworkExecuter(V1Connection):
    def __init__(self, tau=0.02, W=None, B=np.eye(200), C=np.eye(200), sigma=1, kappa=np.pi/4, 
                 alpha=0.9, delta_t=0.001) -> None:
        self.tau = tau
        self.W = W
        self.B = B
        self.C = C
        self.sigma = sigma
        self.delta_t = delta_t
        self.n = len(W)
        self.m = len(B[0])
        self.r_0 = np.zeros(self.n)
        super().__init__(self.m, self.n, kappa, alpha)

    def execute(self, t, orientation, W=None, plot=False):
        if W is not None and len(W) != self.n:
            raise ValueError("Invalid W dimension")
        elif W is not None:
            self.W = W
        rate = self.euler(t, orientation, plot=plot)
        output = self.C @ rate 
        return output + self.sigma * np.random.randn(len(output))

    def get_feedforward_input(self, orientation, contrast=1):
        return contrast * self.cric_gauss(orientation - self.pref)

    def euler(self, t, orientation, plot=False, contrast=1):
        num_iterations = int(np.ceil(t / self.delta_t))
        rate = self.r_0
        h = self.get_feedforward_input(orientation, contrast=contrast)
        Bh = self.B @ h
        for _ in range(num_iterations):
            rate = rate + (self.delta_t / self.tau) * (-rate + self.W @ rate + Bh)
        return rate

def decoder(observations, pref):
    return atan2(np.sum(observations * np.sin(pref)), np.sum(observations * np.cos(pref)))

def decoding_error(predict, actual):
    return np.arccos(np.cos(predict - actual))