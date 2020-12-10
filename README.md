# cnn_mnist_gen_alg
Example of a Convolutional Neural Network applied to the MNIST dataset using Genetic Alogirthm as hyperparameters optimizer
### Model uses GPU to make the training faster to allow several CNN agents to be trained and go over a few generations (genetic alg).
  * cudatoolkit=10.1.243=h74a9793_0
  * cudnn=7.6.5=cuda10.1_0
  * tensorflow=2.1.0=gpu_py37h7db9008_0
  * python=3.7.9=h60c2a47_0
 
### New layers can be added in the "MyCNN" class, thus generating different CNN architecture.
  * The fitness for the GA uses a score that looks for Model Accuracy higher than 50%
  * The GA also increases score if it is trained faster
  * The five best hyperparameters for the models are ploted
  * Since the models weights are starting randomly it is high likely that the same hyperparameters can have multiple accuracy scores.<br/>
**  a fixed starting weights will be provided in a next release.
