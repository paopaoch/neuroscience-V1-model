import numpy as np
import matplotlib.pyplot as plt

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
    

class ImbalanceRingStructureWeights(V1Connection):
    def __init__(self, m=200, kappa=np.pi/4, alpha=100, alpha_inhib=99):
        n = 2 * m
        self.alpha_inhib = alpha_inhib
        super().__init__(m, n, kappa, alpha)

    
    def get_sub_weights(self, alpha):
        pref_matrix = self.pref_diff(self.pref, self.pref)
        W = self.cric_gauss(pref_matrix)
        return self.scale_by_spectral_abisca(W, alpha=alpha)


    def get_weights(self):
        EE = self.get_sub_weights(alpha=self.alpha)
        EI = - self.get_sub_weights(alpha=self.alpha_inhib)
        IE = self.get_sub_weights(alpha=self.alpha)
        II = - self.get_sub_weights(alpha=self.alpha_inhib)
        return np.concatenate((np.concatenate((EE, EI), axis=1),
                                np.concatenate((IE, II), axis=1)), axis=0)


    
def plot_weights(W):
    plt.imshow(W, cmap="seismic", vmin=-np.max(np.abs(np.array(W))), vmax=np.max(np.abs(np.array(W))))
    plt.colorbar()
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.show()



if __name__ == "__main__":
    m=200
    kappa=np.pi/4
    alpha=0.9

    model1 = NoRecurrenceWeights(m, kappa, alpha)
    W1 = model1.get_weights()
    plot_weights(W1)

    model2 = RandomSymmetricConnectivityWeights(m, kappa, alpha)
    W2 = model2.get_weights()
    plot_weights(W2)

    model3 = SymmetricRingStructureWeights(m, kappa, alpha)
    W3 = model3.get_weights()
    plot_weights(W3)

    model4 = BalanceRingStructureWeights(m, kappa, alpha)
    W4 = model4.get_weights()
    plot_weights(W4)