import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# From https://github.com/WojciechMigda/Tsetlini/blob/main/lib/examples/california-housing/src/produce_dataset_alt.py

def _as_bits(x, nbits):
	s = '1' * x + '0' * (nbits - x)
	return np.array([int(c) for c in s])

def _unpack_bits(a, nbits):
	if len(a.shape) > 2:
		raise ValueError("_unpack_bits: input array cannot have more than 2 dimensions, got {}".format(len(a.shape)))

	a = np.clip(a, 0, nbits)
	a_ = np.empty_like(a, dtype=np.uint64)
	np.rint(a, out=a_, casting='unsafe')
	F = np.frompyfunc(_as_bits, 2, 1)
	rv = np.stack(F(a_.ravel(), nbits)).reshape(a.shape[0], -1)
	return rv

class Booleanizer:
	def __init__(self,  max_bits_per_feature = 25):
		self.max_bits_per_feature = max_bits_per_feature

		self.kbd = KBinsDiscretizer(n_bins=max_bits_per_feature+1, encode='ordinal', strategy='quantile')

		return

	def fit(self, X):
		self.kbd_fitted = self.kbd.fit(X)
		
		return

	def transform(self, X):
		X_transformed = self.kbd_fitted.transform(X).astype(int)

		pre = FunctionTransformer(_unpack_bits, validate=False, kw_args={'nbits': self.max_bits_per_feature})
		return pre.fit_transform(X_transformed)