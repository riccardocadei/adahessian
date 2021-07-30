# Discussion on the viability of a modern Second Order Method in Non-Convex Optimization training a Deep Convolutional Neural Network

This repository contains the code and report of the Final Project of [Optimization for Machine Learning (CS-439)](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439) course at [EPFL](https://www.epfl.ch/en/) during Spring term 2021. 

### Team
This project is accomplished by:
- Raphaël Attias: [@raphaelattias](https://github.com/raphaelattias)
- Riccardo Cadei: [@riccardocadei](https://github.com/riccardocadei)
- Milos Novakovic: [@milos-novakovic](https://github.com/milos-novakovic)

### Abstract
Second order algorithms are among the most powerful optimization algorithms with superior convergence properties as compared to first order methods such as SGD and Adam. However computing or approximating the curvature matrix can be very expensive both in per-iteration computation time and memory cost.
In this study we analyze the convenience in using a state-of-the-art Second Order Method (AdaHessian) in Non-Convex Optimization training a Deep Convolutional Neural Network (ResNet18) on MNIST database comparing with traditional First Order Methods. In fact almost all the theoretical results of these methods cannot be extended to Non-Convex optimization and we have to limit to experimental comparisons.
Advantages and disadvantages of both the methods are discussed and a final hybrid method combining the advantages of both is proposed.

For further information about this project, read [`report.pdf`](https://github.com/riccardocadei/CS-439-Optimization-for-ML/blob/main/report.pdf).

### Environment
The project has been developed and test with `python3.8.3`.

Required libraries: 
- `torch` (version 1.9.0.)
- `matplotlib.pyplot` (version 3.3.4.)
- `time` (build in the interpreter)
- `sklearn` (version 0.24.1.)

* * *
 
