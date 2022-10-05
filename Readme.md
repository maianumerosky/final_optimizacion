If you don't want to mess up your base python installation and library dependencies:
- Create a virtual environment:
```
python3 -m venv .env-optimizacion
```
or
```
conda create --name env-optimizacion
```
- Activate it
```
source .env-optimizacion/bin/activate
```
or
```
conda activate env-optimizacion
```
- Install `requirements.txt`:
```
pip install -r requirements.txt
```
or
```
conda install --file requirements.txt
```
- Run jupyter notebook:
```
jupyter notebook
```
---
The `util.py` file has all that you need to experiment with the algorithms. More details on certain things:

`u[i]` and `v[i]` are fixed vectors that satisfy the following conditions:

- u_1, ..., u_4 are four directions in S1 such that the X-rays of any planar convex body in these directions determine it uniquely among all planar convex bodies.
- v_i in S1 orthogonal to u_i such that {u_i, v} is oriented in the same way as the usual orthonormal basis {e1,e2} for R2.


The class `UnknownObject` allows you to experiment with an ellipse. You can generate its noisy measurements like so:

```
ellipse = UnknownObject(alpha=0.05)
measurments = ellipse.noisy_measurements(k=100)
```
alpha is optional (default=0.01).

The classes `Experiment` and `Result` allow you to run experiments with a shared interface. Create a new file with your optimizer and have fun.
 
Example:

```
def example_optimizer(M, s_0, max_iter, std_dev):
    ...

if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        k=100
        M = potato.noisy_measurements(k)
        s_0 = random_s(k)
        params = [M, 200, 50, s_0, 0.1]
        experiment = Experiment(example_optimizer, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))
```