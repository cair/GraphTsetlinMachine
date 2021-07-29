from setuptools import *

setup(
   name='PyTsetlinMachineCUDA',
   version='0.1.8',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='https://github.com/cair/pyTsetlinMachineCUDA/',
   license='MIT',
   description='Massively Parallel and Asynchronous Architecture for Logic-based AI.',
   long_description='Massively Parallel and Asynchronous Architecture for Logic-based AI. Implements the Multiclass Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and multiple layers.',
   keywords ='pattern-recognition cuda machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic tsetlin-machine regression convolution classification multi-layer',
   packages=['PyTsetlinMachineCUDA']
)
