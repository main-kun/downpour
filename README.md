# Downpour SGD

A Scala implementation of [Downpour SGD](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) (Dean et al., 2012) — an asynchronous distributed stochastic gradient descent algorithm. Demonstrated on the MNIST handwritten digit dataset.

## Architecture

The system uses [Akka](https://akka.io/) actors to model the Downpour SGD topology:

- **Master** — orchestrates the system, partitions training data across shards
- **ParameterServer** — holds shared network weights and biases, applies gradient updates from replicas
- **DataShard** — owns a partition of the training data, generates shuffled mini-batches per epoch
- **Replica** — fetches current parameters, computes gradients via backpropagation, pushes updates to the parameter server
- **Evaluator** — periodically evaluates accuracy on the test set and writes results to CSV

The neural network is a fully connected feedforward net (784 → 30 → 10) with sigmoid activations, trained with MSE cost.

## Requirements

- JDK 17+
- [sbt](https://www.scala-sbt.org/)

```
brew install openjdk@17 sbt
```

## Usage

```
sbt compile
sbt run
```

MNIST data is automatically downloaded to `~/.cache/mnist/` on first run.

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `PARALLEL_FACTOR` | `2` | Number of parallel data shards / replicas |
| `NUM_EPOCHS` | `30` | Number of training epochs |
| `MNIST_PATH` | `~/.cache/mnist` | Path to MNIST data directory |

Example:

```
PARALLEL_FACTOR=4 NUM_EPOCHS=50 sbt run
```
## References

- [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) — Dean et al., NIPS 2012
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen
