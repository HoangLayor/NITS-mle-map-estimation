import sys
sys.path.append(".")

from src.mle import GaussianMLE

estimator = GaussianMLE()
estimator.fit([2.1, 1.9, 2.3, 2.0, 2.2])

print("mu =", estimator.mu)
print("sigma =", estimator.sigma)
