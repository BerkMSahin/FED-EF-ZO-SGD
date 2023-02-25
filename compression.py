import numpy as np

compressors = {
    0: 'top',
    1: 'rand',
    2: 'dropout-biased',
    3: 'dropout-unbiased',
    4: 'qsgd'
}


class Compression:
    def __init__(self, num_bits: int,
                 quantization_function: str,
                 dropout_p: float,
                 fraction_coordinates: float):

        self.quantization_function = quantization_function
        self.num_bits = num_bits
        self.fraction_coordinates = fraction_coordinates
        self.dropout_p = dropout_p

    def quantize(self, x):
        if self.quantization_function == 'full':
            return x
        elif self.quantization_function == 'top':
            # Picks the largest k elements.
            q = np.zeros_like(x)
            k = round(self.fraction_coordinates * q.shape[0])
            for i in range(0, q.shape[1]):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q
        elif self.quantization_function == 'rand':
            # Picks the random k elements.
            q = np.zeros_like(x)
            k = round(self.fraction_coordinates * q.shape[0])
            for i in range(0, q.shape[1]):
                perm_i = np.random.permutation(q.shape[0])
                q[perm_i[0:k], i] = x[perm_i[0:k], i]
            return q
        elif self.quantization_function == 'dropout-biased':
            q = np.zeros_like(x)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = x[:, i] * bin_i
            return q
        elif self.quantization_function == 'dropout-unbiased':
            q = np.zeros_like(x)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = x[:, i] * bin_i
            return q / p
        elif self.quantization_function == 'qsgd':
            q = np.zeros_like(x)
            bits = self.num_bits
            s = 2 ** bits
            tau = 1 + min((np.sqrt(q.shape[0]) / s), (q.shape[0] / (s ** 2)))

            for i in range(0, q.shape[1]):
                unif_i = np.random.rand(q.shape[0], )
                x_i = x[:, i]
                if np.linalg.norm(x_i) == 0:
                    continue

                q[:, i] = ((np.sign(x_i) * np.linalg.norm(x_i))
                           / (s * tau)) * np.floor((s * np.abs(x_i) / np.linalg.norm(x_i)) + unif_i)
            return q
        else:
            raise NotImplementedError
