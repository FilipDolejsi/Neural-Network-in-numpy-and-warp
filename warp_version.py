import warp as wp

wp.init()

@wp.kernel
def relu(X: wp.array1d(dtype=wp.float32), out: wp.array1d(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(0.0, X[i])

@wp.kernel
def relu_deriv(X: wp.array1d(dtype=wp.float32), out: wp.array1d(dtype=wp.float32)):
    i = wp.tid()
    out[i] = 1.0 if X[i] > 0 else 0

@wp.kernel
def softmax(Z: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    col = wp.tid()
    num_classes = Z.shape[0]

    z_max = Z[0, col]
    for c in range(num_classes):
        z_max = wp.max(z_max, Z[c, col])

    z_sum = float(0.0)
    for c in range(num_classes):
        z_sum += wp.exp(Z[c, col] - z_max)

    for c in range(num_classes):
        out[c, col] = wp.exp(Z[c, col] - z_max) / z_sum
