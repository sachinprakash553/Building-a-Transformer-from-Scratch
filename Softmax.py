import numpy as np
def softmax(x, axis=-1):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    x : ndarray
        Input data.
    axis : int
        Axis along which the softmax is computed. Default is -1 (last axis).

    Returns
    -------
    ndarray
        Softmax of the input data along the specified axis.
    """
    # Step 1: Subtract the max value from each element for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    # Step 2: Exponentiate the stabilized values
    e_x = np.exp(x - x_max)
    # Step 3: Sum the exponentiated values along the specified axis
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)

    return e_x / sum_e_x

print("Softmax function defined successfully.")
x = np.array([[1.0, 2.0, 3.0],
              [2.0, 4.0, 6.0]], dtype=np.float32)

print("Input:\n", x)
print("Softmax output:\n", softmax(x, axis=-1))

