clear
clc
X=2*rand(20,2)-1;
Y=sign(sin(X(:,1))+X(:,2));
type='classfication';
L_fold=10;
model = initlssvm(X,Y,type,[],[],'RBF_kernel');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{L_fold,'misclass'});
model = trainlssvm(model);
plotlssvm(model);