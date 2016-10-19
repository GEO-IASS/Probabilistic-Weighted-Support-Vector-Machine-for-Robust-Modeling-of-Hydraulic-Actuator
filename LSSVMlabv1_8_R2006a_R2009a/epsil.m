clear
clc
x = (-5:.07:5)';
epsilon = 0.15;
sel = rand(length(x),1)>epsilon;
y = sinc(x)+sel.*normrnd(0,.1,length(x),1)+(1-sel).*normrnd(0,2,length(x),1);
sel'
plot(x,y)