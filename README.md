# Classifying-Cats-vs-Dogs
Uses deep CNN's to classify images of cats and dogs

Training and testing data sets can be found on Kaggle : Dogs vs Cats: Kernel Edition

Framework Used: Theano (with CUDA and CuDNN acceleration)

Architecture: Input -> Conv -> Pool -> Conv -> Pool -> Conv -> FC -> Softmax

SGD with Momentum (decaying with initial = 0.9) used
Decaying learning rate by 5% after every 10 epochs

More information on architecture and implementation provided as comments in the code




