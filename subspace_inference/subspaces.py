"""
    subspace classes
    RandomSpace: Random Subspace
    PCASpace: PCA subspace
"""
import abc

import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize


class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            self.subspace = torch.randn(rank, num_parameters)

        if method == 'fastfood':
            raise NotImplementedError("FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def collect_vector(self, vector):
        pass
    
    def get_space(self):
        return self.subspace


@Subspace.register_subclass('pca')
class PCASpace(Subspace):
    def __init__(self, num_parameters, AS_rank=20, grad_norm = False):
        super(PCASpace, self).__init__()

        self.num_parameters = num_parameters

        assert(isinstance(AS_rank, int))
        assert 1 <= AS_rank

        self.AS_rank = AS_rank

        self.grad_norm = grad_norm
    
    # random subspace is independent of data
    def collect_vector(self, vector):
        pass

    def get_space(self, grads):
       
        AS_rank = max(1, min(self.AS_rank, np.min(grads.shape)))
        print("Number of PCs in Active Subspace: "+str(AS_rank) + ", number of grad samples: "+str(grads.shape[0]))

        if not self.grad_norm:
            U, Sigma, Vt = randomized_svd(grads, n_components=AS_rank, n_iter=5, random_state=1)
#             return Sigma, torch.FloatTensor(Sigma[:, None] * Vt)
            return Sigma, torch.FloatTensor(Vt)
#         else:
#             grads_normalized = normalize(grads, norm='l2')
#             U, Sigma, Vt = randomized_svd(grads_normalized, n_components=AS_rank, n_iter=5, random_state=1)
#             # U, Sigma, Vt = randomized_svd(grads, n_components=AS_rank, n_iter=5, random_state=1)
#             return Sigma, torch.FloatTensor(Sigma[:, None] * Vt)
# #             return Sigma, torch.FloatTensor(Vt)