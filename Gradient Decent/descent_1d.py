def gradient_descent_1d(f, df, x0, alpha=0.1, max_steps=500, tol=1e-6):
    x = x0
    print("t\t x\t\t f(x)\t\t f'(x)\t\t h\t\t Δf")
    for t in range(max_steps):
        fx = f(x)
        dfx = df(x)
        step_size = -alpha * dfx
        x_new = x + step_size
        fx_new = f(x_new)

        print(f"{t}\t {x:.6f}\t{fx:.6f}\t{dfx:.6f}\t{step_size:.6f}\t{(fx_new - fx):.6f}")

        # If our slope is basically 0, we must have found the min.
        if abs(dfx) < tol:
            break

        x = x_new

    return x


def demo_mse_slope():
    print("\n MSE (learn slope m in y = mx)")

    xs = [1, 2, 3, 4]
    ys = [2, 4, 6, 8]
    n = len(xs)

    # Calculate loss function.
    def mse(m):
        return sum((m * xi - yi) ** 2 for xi, yi in zip(xs, ys)) / n

    # Calculate derivative to find the slope of the loss function.
    def dmse(m):
        return (2 / n) * sum((m * xi - yi) * xi for xi, yi in zip(xs, ys))


    m_hat = gradient_descent_1d(mse, dmse, x0=0, alpha=0.1)
    print("learned m:", m_hat)


def demo_quadratic():
    print("Minimize f(x) = (x-5)")

    def f(x):
        return (x - 5) ** 2

    def df(x):
        return 2 * (x - 5)

    final_x = gradient_descent_1d(f, df, x0=0, alpha=0.1)
    print("Final x:", final_x)


if __name__ == "__main__":
    demo_mse_slope()
    demo_quadratic()