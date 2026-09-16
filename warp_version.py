import warp as wp
from matplotlib import pyplot as plt
from numpy import dtypes
from tqdm import tqdm

from abstract_neural_network import AbstractNeuralNetwork

wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"


@wp.func
def relu(X: wp.float32) -> wp.float32:
    return wp.max(0.0, X)


@wp.func
def relu_deriv(X: wp.float32) -> wp.float32:
    return 1.0 if X > 0.0 else 0.0


@wp.kernel
def softmax(Z: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    num_classes = Z.shape[0]

    z_max = Z[0, i]
    for c in range(num_classes):
        z_max = wp.max(z_max, Z[c, i])

    z_sum = wp.float32(0.0)
    for c in range(num_classes):
        z_sum += wp.exp(Z[c, i] - z_max)

    for c in range(num_classes):
        out[c, i] = wp.exp(Z[c, i] - z_max) / z_sum


@wp.kernel
def init_params(seed: wp.int32, param: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * param.shape[1] + j)
    param[i, j] = wp.randf(state=state) - 0.5


@wp.kernel
def forward(
    X: wp.array2d(dtype=wp.float32),
    w_1: wp.array2d(dtype=wp.float32),
    b_1: wp.array2d(dtype=wp.float32),
    w_2: wp.array2d(dtype=wp.float32),
    b_2: wp.array2d(dtype=wp.float32),
    z_1_out: wp.array2d(dtype=wp.float32),
    a_1_out: wp.array2d(dtype=wp.float32),
    z_2_out: wp.array2d(dtype=wp.float32),
    a_2_out: wp.array2d(dtype=wp.float32),
):
    j = wp.tid()
    for h in range(w_1.shape[0]):
        z_1_dotted = wp.float32(0.0)
        for pixel in range(X.shape[0]):
            z_1_dotted += w_1[h, pixel] * X[pixel, j]
        z_1_out[h, j] = z_1_dotted + b_1[h, 0]
        a_1_out[h, j] = relu(z_1_out[h, j])

    for h in range(w_2.shape[0]):
        z_2_dotted = wp.float32(0.0)
        for pixel in range(a_1_out.shape[0]):
            z_2_dotted += w_2[h, pixel] * a_1_out[pixel, j]
        z_2_out[h, j] = z_2_dotted + b_2[h, 0]


@wp.kernel
def one_hot(
    y: wp.array1d(dtype=wp.int32), one_hot_encoding: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()
    one_hot_encoding[i, y[i]] = 1.0


@wp.kernel
def dz2(
    a_2: wp.array2d(dtype=wp.float32),
    one_hot_y: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    out[i, j] = a_2[i, j] - one_hot_y[i, j]


@wp.kernel
def dw2(
    dZ_2: wp.array2d(dtype=wp.float32),
    a_1: wp.array2d(dtype=wp.float32),
    m: wp.int32,
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    acc = wp.float32(0.0)
    for k in range(m):
        acc += dZ_2[i, k] * a_1[j, k]
    out[i, j] = acc / wp.float32(m)


@wp.kernel
def db2(
    dZ_2: wp.array2d(dtype=wp.float32), m: wp.int32, out: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()
    acc = wp.float32(0.0)
    for k in range(m):
        acc += dZ_2[i, k]
    out[i, 0] = acc / wp.float32(m)


@wp.kernel
def dz1(
    w_2: wp.array2d(dtype=wp.float32),
    dZ_2: wp.array2d(dtype=wp.float32),
    z_1: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    h, j = wp.tid()
    acc = wp.float32(0.0)
    for c in range(w_2.shape[0]):
        acc += w_2[c, h] * dZ_2[c, j]
    out[h, j] = acc * relu_deriv(z_1[h, j])


@wp.kernel
def dw1(
    dZ_1: wp.array2d(dtype=wp.float32),
    x: wp.array2d(dtype=wp.float32),
    m: wp.int32,
    out: wp.array2d(dtype=wp.float32),
):
    h, p = wp.tid()
    acc = wp.float32(0.0)
    for j in range(m):
        acc += dZ_1[h, j] * x[p, j]
    out[h, p] = acc / wp.float32(m)


@wp.kernel
def db1(
    dZ_1: wp.array2d(dtype=wp.float32), m: wp.int32, out: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()
    acc = wp.float32(0.0)
    for k in range(m):
        acc += dZ_1[i, k]
    out[i, 0] = acc / wp.float32(m)


@wp.kernel
def update_kernel(
    param: wp.array2d(dtype=wp.float32),
    grad: wp.array2d(dtype=wp.float32),
    alpha: wp.float32,
):
    i, j = wp.tid()
    param[i, j] = param[i, j] - alpha * grad[i, j]


@wp.kernel
def loss_kernel(
    a_2: wp.array2d(dtype=wp.float32),
    y: wp.array1d(dtype=wp.int32),
    out: wp.array1d(dtype=wp.float32),
):
    m = y.shape[0]
    total = wp.float32(0.0)
    for k in range(m):
        total += wp.log(a_2[y[k], k] + 1.0e-9)
    out[0] = -total / wp.float32(m)


@wp.kernel
def predictions_kernel(
    a_2: wp.array2d(dtype=wp.float32), out: wp.array1d(dtype=wp.int32)
):
    j = wp.tid()
    num_classes = a_2.shape[0]
    best_c = wp.int32(0)
    best_val = a_2[0, j]
    for c in range(num_classes):
        if a_2[c, j] > best_val:
            best_val = a_2[c, j]
            best_c = c
    out[j] = best_c


def get_predictions(a_2):
    m = a_2.shape[1]
    predictions = wp.zeros(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel=predictions_kernel,
        dim=m,
        inputs=[a_2],
        outputs=[predictions],
        device=device,
    )
    return predictions


def get_accuracy(y_pred, y):
    y_pred_np = y_pred.numpy() if hasattr(y_pred, "numpy") else y_pred
    y_np = y.numpy() if hasattr(y, "numpy") else y
    return (y_pred_np == y_np).sum() / y_np.size


class WarpNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.w_1, self.b_1, self.w_2, self.b_2 = self.init_params()

    def init_params(self):
        w_1 = wp.zeros((10, 28 * 28), dtype=wp.float32, device=device)
        b_1 = wp.zeros((10, 1), dtype=wp.float32, device=device)
        w_2 = wp.zeros((10, 10), dtype=wp.float32, device=device)
        b_2 = wp.zeros((10, 1), dtype=wp.float32, device=device)
        for idx, p in enumerate([w_1, b_1, w_2, b_2]):
            wp.launch(
                kernel=init_params,
                dim=p.shape,
                inputs=[42 + idx, p],
                device=device,
            )
        return w_1, b_1, w_2, b_2

    def forward(self, X: wp.array2d(dtype=wp.float32)):
        m = X.shape[1]
        z_1 = wp.zeros((10, m), dtype=wp.float32, device=device)
        z_2 = wp.zeros((10, m), dtype=wp.float32, device=device)

        a_1 = wp.zeros((10, m), dtype=wp.float32, device=device)
        a_2 = wp.zeros((10, m), dtype=wp.float32, device=device)

        wp.launch(
            kernel=forward,
            dim=X.shape[1],
            inputs=[X, self.w_1, self.b_1, self.w_2, self.b_2],
            outputs=[z_1, a_1, z_2, a_2],
            device=device,
        )

        wp.launch(
            kernel=softmax,
            dim=z_2.shape[1],
            inputs=[z_2],
            outputs=[a_2],
            device=device,
        )

        return z_1, a_1, z_2, a_2

    def one_hot(self, y: wp.array1d(dtype=wp.int32), num_classes: wp.int32):
        one_hot_encoding = wp.zeros(
            (y.size, num_classes), dtype=wp.float32, device=device
        )

        wp.launch(
            kernel=one_hot,
            dim=y.shape,
            inputs=[y, one_hot_encoding],
            device=device,
        )

        return one_hot_encoding.transpose()

    def backward(self, x, y, a_1, a_2, z_1):
        m = y.size
        one_hot_y = self.one_hot(y=y, num_classes=10)

        dZ_2 = wp.zeros((10, m), dtype=wp.float32, device=device)
        wp.launch(kernel=dz2, dim=(10, m), inputs=[a_2, one_hot_y, dZ_2], device=device)

        dW_2 = wp.zeros((10, 10), dtype=wp.float32, device=device)
        wp.launch(kernel=dw2, dim=(10, 10), inputs=[dZ_2, a_1, m, dW_2], device=device)

        db_2 = wp.zeros((10, 1), dtype=wp.float32, device=device)
        wp.launch(kernel=db2, dim=10, inputs=[dZ_2, m, db_2], device=device)

        dZ_1 = wp.zeros((10, m), dtype=wp.float32, device=device)
        wp.launch(
            kernel=dz1, dim=(10, m), inputs=[self.w_2, dZ_2, z_1, dZ_1], device=device
        )

        dW_1 = wp.zeros((10, x.shape[0]), dtype=wp.float32, device=device)
        wp.launch(
            kernel=dw1, dim=(10, x.shape[0]), inputs=[dZ_1, x, m, dW_1], device=device
        )

        db_1 = wp.zeros((10, 1), dtype=wp.float32, device=device)
        wp.launch(kernel=db1, dim=10, inputs=[dZ_1, m, db_1], device=device)

        return dW_1, db_1, dW_2, db_2

    def update(self, dW_1, db_1, dW_2, db_2, alpha):
        for param, grad in [
            (self.w_1, dW_1),
            (self.b_1, db_1),
            (self.w_2, dW_2),
            (self.b_2, db_2),
        ]:
            wp.launch(
                kernel=update_kernel,
                dim=param.shape,
                inputs=[param, grad, alpha],
                device=device,
            )
        return self.w_1, self.b_1, self.w_2, self.b_2

    def gradient_descent(self, X, y, iterations, alpha):
        print("-" * 20 + "NVIDIA WARP Neural Network TRAINING" + "-" * 20)
        train_loss = []
        X_wp = wp.from_numpy(X, dtype=wp.float32, device=device)
        y_wp = wp.array(y, dtype=wp.int32, device=device)
        pbar = tqdm(range(iterations), desc="training")
        for i in pbar:
            z_1, a_1, z_2, a_2 = self.forward(X=X_wp)
            dW_1, db_1, dW_2, db_2 = self.backward(
                x=X_wp, y=y_wp, a_1=a_1, a_2=a_2, z_1=z_1
            )
            w_1, b_1, w_2, b_2 = self.update(
                dW_1=dW_1, db_1=db_1, dW_2=dW_2, db_2=db_2, alpha=alpha
            )

            acc = get_accuracy(get_predictions(a_2), y_wp)
            loss_out = wp.zeros(1, dtype=wp.float32, device=device)
            wp.launch(
                kernel=loss_kernel, dim=1, inputs=[a_2, y_wp, loss_out], device=device
            )
            loss = loss_out.numpy()[0]

            train_loss.append((i, loss))

            pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{loss:.3f}")
        iters, total_loss = zip(*train_loss)
        plt.plot(iters, total_loss, label='NVIDIA Warp Version')
        plt.ylabel("Train Loss")
        plt.xlabel("Iterations")
        plt.legend()
        plt.savefig("train_loss.png")
        print("-" * 60)
        return w_1, b_1, w_2, b_2

    def test_eval(self, X, y):
        X_wp = wp.from_numpy(X, dtype=wp.float32, device=device)
        y_wp = wp.array(y, dtype=wp.int32, device=device)
        _, _, _, test_forward = self.forward(X=X_wp)
        acc = get_accuracy(get_predictions(test_forward), y_wp)
        loss_out = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[test_forward, y_wp, loss_out],
            device=device,
        )
        loss = loss_out.numpy()[0]
        print("-" * 20 + "NVIDIA WARP Neural Network TESTING" + "-" * 20)
        print("Test Accuracy:", acc)
        print("Test Loss:", loss)
        print("-" * 60)
        return acc, loss
