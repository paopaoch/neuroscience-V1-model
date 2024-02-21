import numpy as np
from network_builder import V1Connection
import matplotlib.pyplot as plt
from math import atan2

class NetworkExecuter(V1Connection):
    def __init__(self, tau=0.02, W=None, B=np.eye(200), C=np.eye(200), sigma=1, kappa=np.pi/4, alpha=0.9, delta_t=0.001) -> None:
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

        self.orientations = np.linspace(0, 2*np.pi, 20)
        self.contrasts = np.linspace(0, 1, 20)

    
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
        rates = []
        
        if plot:
            neuron_index = int(self.n / (2 * np.pi / orientation))

        for _ in range(num_iterations):
            rate = rate + (self.delta_t / self.tau) * (-rate + self.W @ rate + Bh)
            
            if plot:
                rates.append(rate[neuron_index])

        if plot:
            x = np.linspace(0,t, num_iterations)
            plt.plot(x, rates)
            plt.show()

        return rate


    def run_all_orientation_and_contrast(self):
        tuning_curve_2d = []
        for contrast in self.contrasts:
            tuning_curve_1d = []
            for orientation in self.orientations:
                tuning_curve_1d.append(self.euler(t=0.2, orientation=orientation, contrast=contrast))
            tuning_curve_2d.append(tuning_curve_1d)
        return tuning_curve_2d


def decoder(observations, pref):
    return atan2(np.sum(observations * np.sin(pref)), np.sum(observations * np.cos(pref)))


def decoding_error(predict, actual):
    return np.arccos(np.cos(predict - actual))


if __name__ == "__main__":
    from network_builder import SymmetricRingStructureWeights, BalanceRingStructureWeights, RandomSymmetricConnectivityWeights, NoRecurrenceWeights

    orientation = np.pi
    tau = 0.02
    m=200
    kappa=np.pi/4
    alpha=0.9
    B = np.eye(m)
    C = np.eye(m)
    sigma = 1
    model = SymmetricRingStructureWeights(m, kappa, alpha)
    W = model.get_weights()
    ne = NetworkExecuter(tau, W, B, C, sigma, kappa, alpha)

    noisy_output = ne.execute(2, orientation, plot=True)
    plt.plot(ne.pref, noisy_output)
    plt.show()


    # B = np.eye(2*m, m)
    # C = np.eye(m,2*m)
    # model4 = BalanceRingStructureWeights(m, kappa, alpha)
    # W = model4.get_weights()
    # ne = NetworkExecuter(tau, W, B, C, sigma, kappa, alpha)

    # noisy_output = ne.execute(50, orientation)
    # plt.plot(ne.pref, noisy_output)
    # plt.show()

    decoded = decoder(noisy_output, ne.pref)
