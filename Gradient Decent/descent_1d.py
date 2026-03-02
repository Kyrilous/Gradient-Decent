def gradient_decent_1d(f, df, x0, alpha=0.1, steps=10):
    x = x0
    print("t\t x\t\t f(x)\t\t f'(x)\t\t h")
    for t in range(steps):
        fx = f(x)
        dfx = df(x)
        h = -alpha * dfx
        print(f"{t}\t {x:.6f}\t{fx:.6f}\t{dfx:.6f}\t {h:.6f}")
        x = x + h
        if abs(dfx) < 1e-6:
            break
    return x


def f(x):
    return (x-5)**2

def df(x):
    return 2*(x-5)

final_x = gradient_decent_1d(f, df, x0=0, alpha=0.1, steps=10)
print("Final x:", final_x)