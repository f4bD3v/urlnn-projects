function [] = simulate(theta,eta,tao,w,rounds)

%initialization
i = 1:100;
sigma = 10;
%ys = zeros(1,rounds/1000);
thetas = zeros(1,rounds/1000);
weights = zeros(100,rounds/1000);
Fs = zeros(1,rounds/1000);
xs = zeros(5,100);
ytemp = zeros(1,1000);
counter = 1;
for j=1:5
    mu = -10+j*20;
    xs(j,:) = 1/sqrt(2*pi)/sigma*exp(-(i-mu).^2/sigma^2);
end

%working loop
for round = 1:rounds
    x = xs(randi(5),:);
    %ys(round) = max(dot(x,weights(:,round)),0);
    y = max(dot(x,w),0);
    ytemp(counter) = y;
    %dthe = (-thetas(round)+ys(round)^2)/tao;
    dthe = (-theta+y^2)/tao;
    %thetas(round+1) = thetas(round)+dthe; %dt=1
    theta = theta+dthe;
    %disp(ys(round)^2-ys(round)*thetas(round+1))
    %dw = eta*(x*ys(round)^2-x*ys(round)*thetas(round+1));
    dw = eta*(x*y^2-x*y*theta);
    %weights(:,round+1) = max(weights(:,round)+dw',0);
    w = max(w+dw,0);
    %Fs(round) = mean(ys(1:round).^3)/sqrt(mean(ys(1:round).^2));    
    if counter == 1000
        thetas(round/1000) = theta;
        weights(:,round/1000) = w;
        Fs(round/1000) = mean(ytemp.^3)/sqrt(mean(ytemp.^2));
        counter = 0;
    end
    counter = counter + 1;
end


%results

t = linspace(1,rounds,length(thetas));

     j=1;
     y1 = zeros(length(thetas));
     for k = 1:length(thetas)
         y1(k) = dot(xs(j,:),weights(:,k));
     end
j=2;
     y2 = zeros(length(thetas));
     for k = 1:length(thetas)
         y2(k) = dot(xs(j,:),weights(:,k));
     end
     j=3;
     y3 = zeros(length(thetas));
     for k = 1:length(thetas)
         y3(k) = dot(xs(j,:),weights(:,k));
     end
     j=4;
     y4 = zeros(length(thetas));
     for k = 1:length(thetas)
         y4(k) = dot(xs(j,:),weights(:,k));
     end
     j=5;
     y5 = zeros(length(thetas));
     for k = 1:length(thetas)
         y5(k) = dot(xs(j,:),weights(:,k));
     end

plot(t,y1,'b',t,y2,'r',t,y3,'g',t,y4,'y',t,y5,'k',t,thetas,'m',t,Fs,'c','LineWidth',2);

legend('y^1','y^2','y^3','y^4','y^5','theta','F','Location','EastOutside');
xlabel('Time');


%figure, plot(weights(:,rounds))
figure,plot(w);
xlabel('i');
ylabel('w_i');

%w = ys;
%w = weights(:,rounds+1);

end