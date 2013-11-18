function w = randomweights()

mu = 3;
sigma = 1;
i = ones(1,100);
w = normrnd(mu*i,sigma);

end