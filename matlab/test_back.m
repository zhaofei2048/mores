clear all; 
% close all; 
clc;
% 测试粗糙面散射模型

AN = 8;

f = 1e9;
% kdel = 0.6/sqrt(AN);
% kcor = 3.0/sqrt(AN);
% kdel = 0.6;
% kcor = 3.0;
kdel = 1.69;
kcor = 8.4;
epsr = 3.0 - 0.1j;
c = physconst('LightSpeed');
Lambda = c / f;
k = 2 * pi / Lambda;

delta = kdel / k;
corr_len = kcor / k;

theta_is = linspace(0, 80, 80);

N = length(theta_is);
vvs = zeros(N, 1);
hhs = zeros(N, 1);

for i = 1:N
    theta_i = theta_is(i);
    theta_s = theta_i;
    Mue = Fun_R_rough_IEMs(f, [theta_s, 180], [theta_i, 0], 1, epsr, [delta, corr_len], 'exp', 1.5, true);
    vvs(i) = 10*log10(4*pi * Mue(1, 1));
    hhs(i) = 10*log10(4*pi * Mue(2, 2));
end

figure;
plot(theta_is, vvs, "b-", DisplayName="VV"); hold on;
plot(theta_is, hhs, "g--", DisplayName="HH");
grid on;
hold off;