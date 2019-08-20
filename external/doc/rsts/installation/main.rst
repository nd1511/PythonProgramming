Installation
============

You have to take following steps to use ReNom in your environment.

1. Install the python(we confirm the operation in python 2.7, 3.4, 3.5 and 3.6)
2. Install the ReNom environment

First, you have to install the python.
There are many web pages that explain how to intall the python.
And, you can download ReNom from following link.

URL: https://github.com/ReNom-dev-team/ReNom

If you already installed the GPU environments.

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNom.git
   cd ReNom
   python setup.py build_ext -f -i
   pip install -e .

If you did not set the GPU environments.

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNom.git
   cd ReNom
   pip install -e .

**Requirements**

Please refer to the README.md file in the ReNom github page.

https://github.com/ReNom-dev-team/ReNom
