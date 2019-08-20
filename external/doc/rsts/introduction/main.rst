Introduction of version 2
===========================

In ReNom version 2, automatic differentiation feature have been added to
version 1.0. Users are able to build neural network model with a lot of flexibility.

**Concept of version 2**

ReNom 2 is focusing on its usability first, as the same as previous version.

The syntax of ReNom version 2 is aligned to NumPy, so that users can compute
differential value adding a tiny script change to the formula written in NumPy style.

By reducing user interfaces, ReNom became a NumPy user friendly library package while
enables users to build a neural network model more flexibly.

Following is a comparison of NumPy and ReNom coding style.

● Numpy
   .. code-block:: python

      >>> import numpy as np
      >>> a, b = np.arange(2), np.arange(2)
      >>> x = np.arange(2)
      >>> z = np.sum(a*x + b)
      >>> print(z)
      2.0

● ReNom
   .. code-block:: python

      >>> import numpy as np
      >>> import renom as rm
      >>> a, b = np.arange(2), np.arange(2)
      >>> x = rm.Variable(np.arange(2))
      >>> z = rm.sum(a*x + b)
      >>> print(z)
      2.0

      >>> dx = z.grad().get(x)
      >>> print(dx)
      [0, 1]

Like this, ReNom users can compute gradient by changing only a few NumPy code.

**Auto Differentiation**

In ReNom, users can create calculation graph with a simple step.
First, defining differentiation target variable as Variable,
then scripting formula as the same syntax as NumPy.

   .. code-block:: python

      >>> import renom as rm
      >>> a, b = 2, 3
      >>> x = rm.Variable(1)
      >>> z = a*x + b
      >>> gradient = z.grad().get(x)
      >>> print(gradient)
      2.0

Variable class is inherited ndarray class of NumPy[ref],
users can create/build/establish calculation graph similar way to NumPy.

**Sequential Model**

As the same as previous ReNom versions, users can define the model, simply piling the layers up.

   .. code-block:: python

      import renom as rm

      model = rm.Sequential([
            rm.Dense(100),
            rm.Relu(),
            rm.Dense(10)
         ])

In ReNom, defined class names are capitalized. As mentioned,
Sequential model can be instantiated by providing a layer object list.


**Functional Model**

In ReNom 2, some layers previously regarded as objects such as Activation function layer,
fully connected layer are able to be handled functionally.

   .. code-block:: python

      import renom as rm

      class NN(rm.Model):

         def __init__(self):
            self._layer1 = rm.Dense(100)
            self._layer2 = rm.Dense(10)

         def forward(self, x):
            h = rm.relu(self._layer1(x))
            z = rm._layer2(h)
            return z

      model = NN()

In ReNom, defined function names are small lettered.
As above, defined functions are able to handle layer objects.


**Computation with GPU**

In order to use GPU, users need to install Cuda-Toolkit and cuDNN.
To switch GPU on/off, simply call following function.

   .. code-block:: python

      import renom as rm
      rm.set_cuda_active(True)

