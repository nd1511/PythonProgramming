import numpy as np


class GradientClipping(object):
    """
    This class allows gradient clipping.
    """

    def __init__(self, threshold=0.5, norm=2):
        self.threshold = threshold
        self.norm = norm

    def __call__(self, gradient=None):
        """
        This function clips the gradient if gradient is above threshold.
        This function will appear in Grad class of ReNom in the near future.
        The calculation is dones as shown below:

        .. math::

            \hat{g} \leftarrow \frac{\partial \epsilon}{\partial \theta} \\
            if ||\hat{g}|| \geq {\it threshold} \hspace{5pt} {\bf then}\\
            \hat{g} \leftarrow \frac{threshold}{||\hat{g}||}\hat{g} \\

        Args:
            gradient: gradient object
            threshold(float): theshold
            norm(int): norm of gradient

        Examples::
            >>> from **** import GradientClipping
            >>> grad_clip = GradientClipping(threshold=0.5,norm=2)
            >>>
            >>> grad = loss.grad()
            >>> grad_clip(grad)
            >>>
            >>> grad.update(Sgd(lr=0.01))
        """

        threshold = self.threshold
        norm = self.norm

        assert gradient is not None, "insert the gradient of model (model.grad())"

        # setting variables etc.
        variables = gradient.variables
        norm = float(norm)
        threshold = float(threshold)

        if norm == float("inf"):
            # h infinity
            total_norm = np.max([np.max(i) for i in np.max(variables.values())])
        else:
            # regular norm
            total_norm = 0
            for i in variables:
                arr = variables[i]**norm
                total_norm += arr.sum()
            total_norm = total_norm ** (1 / total_norm)

        # process gradient
        if threshold < total_norm:

            for i in variables:
                variables[i] = threshold * variables[i] / (total_norm + 1e-6)
