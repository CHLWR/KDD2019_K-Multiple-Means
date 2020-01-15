function [X, y] = face_gen(num, noise)

n1=floor(0.05*num);
n3=floor(0.15*num);
n4=floor(0.75*num);

interval1=1.25; interval2=1.25;
% left eye
m = [-interval1,interval2];
C = 0.5*noise*eye(2);
x1 = mvnrnd(m,C,n1);
y1 = 1+zeros(n1,1);
% right eye
m = [interval1,interval2];
x2 = mvnrnd(m,C,n1);
y2 = 2+zeros(n1,1);

% nose 
x3=zeros(n3,2);
r=1.5;
t=(5/4*pi):pi/(2*n3-1):(7/4*pi); % ÏÂ°ëÔ²
x3(:, 1) = r.*cos(t)'+randn(n3,1)*noise;
x3(:, 2) = r.*sin(t)'+randn(n3,1)*noise;
y3 = 3+zeros(n3,1);

% face
upright = 0.5;
x4=zeros(n4,2);
r = 3;
curve = 2.5;
t = unifrnd(0,0.8,[1,n4]);
x4(:, 1) = r.*sin(curve*pi*t) + noise*randn(1,n4);
x4(:, 2) = r.*cos(curve*pi*t) + noise*randn(1,n4)+upright;;
y4 = 4+zeros(n4,1);

X = [x1;x2;x3;x4];
y = [y1;y2;y3;y4];
