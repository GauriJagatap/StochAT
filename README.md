Scripts and notebooks for experiments on ATENT.

notebooks/ folder contains [corresponding figures and tables]:
1. cifar10_exps_linf.ipynb (CIFAR10 experiments on WideResNet34 for L-Inf attacks) [Table 3 and Figure 3]
2. cifar10_exps_l2.ipynb (CIFAR10 experiments on ResNet18 or ResNer20 for L-2 attacks) [Table 4 and Table 5]
3. mnist_exps_linf.ipynb (MNIST experiments on small CNN for L-Inf attacks) [Table 2]
4. mnist_exps_l2.ipynb (MNIST experiments on LeNet5 for L-2 attacks) [Table 1]
5. ATENT_as_attack.ipynb (script that runs ATENT as an attack on pretrained model) [Table 6]
6. twoclass_decision_boundary.ipynb (notebook to generate visualizations of decision boundaries of ATENT, TRADES, PGD, SGD for 2-D classification) [Figure 1]

scripts/ folder contains:
1. attacks.py (containing CW and DeepFool attacks. PGD attacks are already included in all notebooks.)

adversarial/ folder contains:
1. functional.py (containing main module entropySmoothing.py required to generate SGLD updates)



