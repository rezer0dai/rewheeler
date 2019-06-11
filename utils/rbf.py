import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn

class RbfState:
    """ code : https://www.udemy.com/deep-reinforcement-learning-in-python/ ( RBF part ) and also here : https://github.com/dennybritz/reinforcement-learning
        + bring more features to simple state ~ faster/possible learning
        - need to figure out good parameters for problem ( gama, # )
          and also need to sample space ( by env sampling, or by your custom )
    """
    def __init__(self, env, gamas, components, sampler = None):
        observation_examples = sampler(env) if None != sampler else self._sampler(env)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
            (
                "rbf%i"%i,
                RBFSampler(gamma=g, n_components=c)
            ) for i, (g, c) in enumerate(zip(gamas, components))])

        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.size = sum(components)

    def _sampler(self, env):
        return np.array([env.observation_space.sample() for x in range(10000)])

    def transform(self, state):
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized
