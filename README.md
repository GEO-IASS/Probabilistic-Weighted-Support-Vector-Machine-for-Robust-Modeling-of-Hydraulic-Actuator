
"train_robust_weighted_lssvm" function is the train function, 
"sim_robust_weighted_lssvm" function is the simulation function.
in train function, the input feature are describle in the following:
xy_train: the data which are used to train the model;
lanbada: regularization parameter for global weights;
gama: regularization parameter for empirical risk;
cluster_num: the number of the subsets;
resample: the number of data on each subset;
sig_new: the center of RBF function;
window_width: the width of window function.


in simulation function, the input feature are describle in the following:
xy: the data which are used to test;
p: the probability distribution of all local LS-SVM;
xy_all and yita are the middle parameters got from the train function;
sig: the center of RBF function;

the output of simulation function is the predicted result.


the file of "LSSVMlabv1_8_R2006a_R2009a" is the toolbox of LS-SVM, which is necesary for the proposed method.
