Change logs 
============

**2.7.2**

1. Modified BatchNormalize layer for calculating back propagation with inference mode.
2. Fixed bug in Rmsprop.


**2.7.0**

1. Support ONNX(https://onnx.ai/).


2. Add ``GroupConvolution``
    :py:meth:`renom.layers.function.group_conv2d.GroupConv2d`.

3. Add ``DeconvNd``
    :py:meth:`renom.layers.function.deconvnd.DeconvNd`.

4. Add Nesterov's Accelerated Gradient Method to SGD.
    :py:meth:`renom.optimizer.Sgd`.

5. Add ``He Normal`` weight initializer.
    :py:meth:`renom.utility.initializer.HeNormal`.

6. Add ``He Uniform`` weight initializer.
    :py:meth:`renom.utility.initializer.HeUniform`.

