function [Ms, Coeff_test, sca_test] = Fun_R_rough_IEMs(f, angle_s, angle_i, epsilon_1, epsilon_2, sp, SpR, x, shadow_flag)
%FUN_R_ROUGH_IEMs
%使用IEM模型计算向上单次散射的Mueller矩阵（非相干分量,FSA坐标系）
%   来自IEMv1
%   Author: Fei Zhao
%   Create: 2022-11-25
%   Update:
%   INPUT:
%       f : frequency (Hz) of incident wave
%       angle_s : [theta_s, phi_s] (deg) scattering angles (theta_s is angle between
%           reflected wave and positive z axis)
%       angle_i : [theta_i, phi_i] (deg) incident and azimuth angles
%       epsilon_1 real|complex : relative dielectric constant(er-1j*ei) of upper medium 1
%       epsilon_2 real|complex : relative dielectric constant(er-1j*ei) of lower medium 2
%       sp : [delta, corr_len] RMS height (m) and correlation length (m) of rough
%           surface
%       SpR <char>array : one of {'Gauss', 'exp', 'x-power', 'x-exp'}, which is the spectrum of the rough
%           surface, i.e., the Fourier transform of corr function.
%       x scalar : coefficient (>1) needed for 'x-power' and
%           'x-exp(onential)' correl. fnc. (default 1.5)
%       shadow_flag: true for adding shadow function, default is false
%   OUTPUT:
%          Ms   : <real>matrix[4*4] is the Mueller matrix of AIEM single scattering term
%   ref:
%
%   Note: 
%       计算得到的Mueller矩阵为FSA坐标系  
%           ki_ = (s*csf, s*sf, -cs)
%           ks_ = (ss*csfs, ss*sfs, css)
%           hi_ = (-sf, csf, 0)
%           vi_ = (-cs*csf, -cs*sf, -s)
%           hs_ = (-sfs, csfs, 0)
%           vs_ = (css*csfs, css*sfs, -ss)
%                   其中cs = cos(theta_i), s=sin(theta_i), csf = sin(phi_i),
%                   sf = sin(phi_i); css = cos(theta_s), ss = sin(theta_s),
%                   csfs = cos(phi_s), sfs = sin(phi_s).
%
%
%   Outline:
%           - check inputs
%           - variable conversion
%           - Fresnel reflection coefficient: by incident angle | by
%           specular angle | by 0
%           - Kirchhoff field coefficient calculating: fqp
%           - Definition of Complementary field coefficient function
%           - Complementary coefficient calculating: Fqp(u,v,d)
%           - Fresnel reflection coefficient by transition function
%           - update the Kirchhoff coefficient with transition function
%           - Single scattering term calculating
%           - Mueller matrix
%           - Mueller matrix to FSA coordinate
%           - add Shadowing function
%           - debug part
%           - other routines
%   Logs:
%       2023-2-10:
%       确保theta_i和theta_s始终不严格相等，同时发现粗糙尺度过大时，AIEM大入射角计算不正确，而在双战散射计算时不要让入射角处在布儒斯特角附近
%       2023-2-12: IEMFhv1中的B(6)项的负号纠正为正号
%                  为了数值稳定，改为永远不让入射角严格等于散射角，方式是发现相等时给散射角加上0.001度
%       2023-2-13: 过渡函数权重计算调整为始终使用后向散射时的计算方式（权重仅和入射角、介电常数、粗糙度&谱有关，特别注意wvnb不要引入散射角影响）
%                   添加shadow_flag参数控制是否添加阴影遮蔽函数
%       2023-2-18: 修正IEM坐标系到FSA坐标系的转换关系（原来写成了BSA到FSA的转换）

%% check inputs
if ~exist('x', 'var')
    x = 1.5;
end

if ~exist('shadow_flag', 'var')
    shadow_flag = false;
end

if angle_s(1) == -angle_i(1) || angle_s(1) == angle_i(1)   % 永远不让theta_i和theta_s严格相等
    angle_s(1) = angle_s(1) + 0.001; % 确保数值稳定
end

% if angle_s(1) == -angle_i(1) || (angle_s(1)==angle_i(1) && abs(angle_s(2)-angle_i(2))==180)   % backscattering
% %    disp('后向散射');
%     angle_s(1) = angle_s(1) + 0.001; % 确保数值稳定
% end

% 下面的操作用以避免双站散射时散射角等于入射角时出现的不连续性！！！！！！！！！！！
%  引起不连续的原因还不知道！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% if angle_s(1) == angle_i(1)
%     angle_s(1) = angle_s(1) + 0.001; % 确保数值稳定
% end

%% key variables
theta_i = deg2rad(angle_i(1));
phi_i = deg2rad(angle_i(2));
theta_s = deg2rad(angle_s(1));
phi_s = deg2rad(angle_s(2));

er = epsilon_2 / epsilon_1;
mur = 1;

delta = sp(1);
corr_len = sp(2);
c = physconst('Lightspeed');
k1 = real(2 * pi * f * sqrt(1 * epsilon_1)/c);
k2 = real(2 * pi * f * sqrt(1 * epsilon_2)/c);
k1 = 2 * pi * f * sqrt(1 * epsilon_1)/c;
k2 = 2 * pi * f * sqrt(1 * epsilon_2)/c;
k = k1;
% eta_r = sqrt(mur / er); % 相对波阻抗
eta_ist = 1;    % 计算透射场则为eta_r
% k1r = real(k1);
kdel = k1 * delta;  % real?
kcor = k1 * corr_len; % real?
kdel2 = kdel * kdel;


cs = cos(theta_i);
s = sin(theta_i);
css = cos(theta_s);
ss = sin(theta_s);
phi_s = phi_s - phi_i;
phi_i = 0;
csfs = cos(phi_s);
sfs = sin(phi_s);


% 波矢量
kx = k1 * s; ky = 0; kz = k1 * cs;
ksx = k1 * ss * csfs; ksy = k1 * ss * sfs; ksz = k1 * css;
ki_ = [s; 0; -cs];  % 带_的表示矢量
% ks_ = [ss * csfs; ss * sfs; css];
% 入射、散射极化矢量 (IEM coordinate)
h_ = [0; 1; 0];
v_ = [cs; 0; s];
hs_ = [sfs; -csfs; 0];
vs_ = [-css * csfs; -css * sfs; ss];

%% 确定迭代次数
Nmax = 1;
myeps = 1e-32;

err = kdel2;
while(abs(err) > myeps)
    Nmax = Nmax + 1;
    err = err * kdel2 / Nmax;
end

%% 计算粗糙谱Wn
Kx = ksx - kx; Ky = ksy - ky;
wvnb = sqrt(abs(Kx^2 + Ky^2));
[Wn, rss] = Fun_rough_spectrum(SpR, Nmax, delta, corr_len, wvnb, x);

%% Fresnel coefficients
%-----------------------------------------------------------------
% R(theta_i) : reflection coefficients based on the incident angle
%-----------------------------------------------------------------
tmp1 = sqrt(er - s*s);
rvi = (er * cs - tmp1) / (er * cs + tmp1);   % rv = Hv_s / Hv_i = Ev_s / Ev_i
tmp2 = sqrt(mur * er - s*s);
rhi = (mur * cs - tmp2) / (mur * cs + tmp2); % rh = Eh_s / Eh_i
% if er == inf % for PEC
%     rvi = 1;
%     rhi = -1;
% end
rvhi = (rvi - rhi) / 2.0;

%-----------------------------------------------------------------
% R(theta_sp) : reflection coefficients based on the specular angle
%-----------------------------------------------------------------
cssp = sqrt(1 + cs * css - s * ss * csfs)/sqrt(2);
ssp = sqrt(1 - cssp * cssp);
tmp1 = sqrt(er - ssp*ssp);
rvsp = (er * cssp - tmp1) / (er * cssp + tmp1); % rv = Hv_s / Hv_i = Ev_s / Ev_i
tmp2 = sqrt(mur * er - ssp*ssp);
rhsp = (mur * cssp - tmp2) / (mur * cssp + tmp2); % rh = Eh_s / Eh_i
% if er == inf % for PEC
%     rvsp = 1;
%     rhsp = -1;
% end
rvhsp = (rvsp - rhsp) / 2.0;

%-----------------------------------------------------------------
% R(0) : reflection coefficients based on 0 incident angle
%-----------------------------------------------------------------
rv0 = (sqrt(er) - 1.0) / (sqrt(er) + 1.0);
rh0 = -rv0;

%% 计算Kirchhoff系数

% 选择使用何种菲涅尔系数来计算过渡函数
% rv = rvi;
% rh = rhi;
rv = rvsp;
rh = rhsp;
% rv = rv0;
% rh = rh0;

Zxf = - (ksx - kx) / (ksz + kz);
Zyf = - (ksy - ky) / (ksz + kz);
n_ = [-Zxf; -Zyf; 1];   % 局部法向矢量（但不是单位矢量），对于fqp来说n_就是ki_和ks_的角平分线方向
normlized_n_ = n_ / norm(n_);
t_ = cross(ki_, normlized_n_);
normt = norm(t_);
t_ = t_ / normt;  % 单位化

d_ = cross(ki_, t_);
hsnv = dot(hs_, cross(n_, v_)); % =1 when backscattering
hsnh = dot(hs_, cross(n_, h_));
hsnt = dot(hs_, cross(n_, t_));
hsnd = dot(hs_, cross(n_, d_));
vsnh = dot(vs_, cross(n_, h_)); % =-1 when backscattering
vsnv = dot(vs_, cross(n_, v_));
vsnt = dot(vs_, cross(n_, t_));
vsnd = dot(vs_, cross(n_, d_));
hd = dot(h_, d_);
vt = dot(v_, t_);

% 稍后使用过渡函数更新
fvv = -((1-rv) * hsnv + eta_ist * (1 + rv) * vsnh) ...
    - (rh + rv) * vt * (hsnt + eta_ist * vsnd);
fhh = ((1 + rh) * vsnh + eta_ist * (1 - rh) * hsnv) ...
    - (rh + rv) * hd * (vsnd + eta_ist * hsnt);

%% 计算补偿场系数函数
    function F = IEMFvv(u, v, rv)
        rvp = (1 + rv);
        rvm = (1 - rv);
        qi = sqrt(k1^2 - u^2 - v^2);
        qt = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, 1, 0);

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -(rvm / qi - rvp * mur / qt) * rvp * C(1) ...
            + (rvm / qi - rvp / qt) * rvm * C(2) ...
            + (rvm / qi - rvp / er / qt) * rvp * C(3) ...
            + eta_ist * (...
            (rvp / qi - rvm * er / qt) * rvm * C(4) ...
            + (rvp / qi - rvm / qt) * rvp * C(5) ...
            + (rvp / qi - rvm / mur / qt) * rvm * C(6) ...
            );
    end

    function F = IEMFhh(u, v, rh)
        % 在specular角度下计算得到的补偿场系数
        rhp = (1 + rh);
        rhm = (1 - rh);
        qi = sqrt(k1^2 - u^2 - v^2);
        qt = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, 1, 0);

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -(rhp / qi - rhm * mur / qt) * rhm * C(4) ...
            -(rhp / qi - rhm / qt) * rhp * C(5) ...
            -(rhp / qi - rhm / er / qt) * rhm * C(6) ...
            + eta_ist * (...
            (rhm / qi - rhp * er / qt) * rhp * C(1) ...
            - (rhm / qi - rhp / qt) * rhm * C(2) ...
            - (rhm / qi - rhp / mur / qt) * rhp * C(3) ...
            );
    end

    function F = IEMFvh(u, v, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k1^2 - u^2 - v^2);
        qt = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, 1, 0);

        % B
        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F1 = rm / qi * rp * B(4) + rm / qi * rm * B(5) + rm / qi * rp * B(6) ...
            + eta_ist * (rp / qi * rm * B(1) - rp / qi * rp * B(2) - rp / qi * rm * B(3));
        F2 = -rp * mur / qt * rp * B(4) - rp / qt * rm * B(5) - rp / qt / er * rp * B(6) ...
            + eta_ist * (-rm * er / qt * rm * B(1) + rm / qt * rp * B(2) + rm / qt / mur * rm * B(3));

        F = F1 + F2;
    end

    function F = IEMFhv(u, v, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k1^2 - u^2 - v^2);
        qt = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, 1, 0);

        % B
        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F1 = rm / qi * rp * B(1) - rm / qi * rm * B(2) - rm / qi * rp * B(3) ...
            + eta_ist * (rp / qi * rm * B(4) + rp / qi * rp * B(5) + rp / qi * rm * B(6));
        F2 = -rp * mur / qt * rp * B(1) + rp / qt * rm * B(2) + rp / qt / er * rp * B(3) ...
            + eta_ist * (-rm * er / qt * rm * B(4) - rm / qt * rp * B(5) - rm / qt / mur * rm * B(6));

        F = F1 + F2;
    end

%% prepare for factorial calc (avoid overflow)
nksz = ksz / k; % 归一化
nkz = kz / k;
nv = reshape(1:Nmax, [Nmax, 1]);   % n vector
% 计算(k*sigma)^2n / n! = (ksigma2/1) * (ksigma2/2) * (ksigma2/3) * ... * (ksigma2/n) 
kdel2_n = kdel2 ./ nv;
kdel2_n = cumprod(kdel2_n);

%% 菲涅尔反射系数的过渡模型
%-----------------------------------------------------------------
% R_transition : reflection coefficients based on the a transition function
%   by T.D. Wu & A.K. Fung (2001)
%-----------------------------------------------------------------
% todo
%-----transition when backscattering---------------------------------------
Ftv = 8.0 * (rv0^2) * s^2 * (cs + sqrt(er - s^2)) / (cs * sqrt(er - s^2));
Fth = -8.0 * (rh0^2) * s^2 * (cs + sqrt(er - s^2)) / (cs * sqrt(er - s^2));
Spv0 =1.0./((abs(1.0 +8.0 .*rv0./(cs.*Ftv))).^2.0);
Sph0 =1.0./((abs(1.0 +8.0 .*rv0./(cs.*Fth))).^2.0);
sum1 = 0.0;
sum2 = 0.0;
sum3 = 0.0;
temp1 = 1.0;

[Wnt] = Fun_rough_spectrum(SpR, Nmax, delta, corr_len, sqrt((2*k1*s)^2 + 0^2), x);    % Wnt for transition function
for n = 1: Nmax
        fn =n;
        temp1= temp1.*(1./fn);
        sum1= sum1 + temp1.*((kdel.*cs).^(2.0 .*fn)).*Wnt(n);
        sum2= sum2 + temp1.*((kdel.*cs).^(2.0 .*fn))*(abs(Ftv+2.0 .^(fn+2.0 ).*rv0./cs./(exp((kdel.*cs).^2.0 ))).^2.0 ).*Wnt(n);
        sum3= sum3 + temp1.*((kdel.*cs).^(2.0 .*fn))*(abs(Fth+2.0 .^(fn+2.0 ).*rv0./cs.*(exp(-(kdel.*cs).^2.0 ))).^2.0 ).*Wnt(n);
end
sigma_vv_c = (abs(Ftv).^2.0 ).*sum1;
sigma_vv_total = sum2;
sigma_hh_c = (abs(Fth).^2.0 ).*sum1;
sigma_hh_total = sum3;
%--------------------------------------------------------------------------

%--------------transition by IEM-------------------------------------------
% % calc Spv0 & Sph0
% % rv = rvi; rh = rhi;
% Fvv = IEMFvv(-kx, -ky, rv);    % 用IEM的看看！！！！
% Fvvs = IEMFvv(-ksx, -ksy, rv);
% Av = abs(0.5 * (css * Fvv + cs * Fvvs))^2;
% Bv = (css + cs) * real(fvv * (css * Fvv + cs * Fvvs));
% Cv = abs((css + cs) * fvv)^2;
% 
% Fhh = IEMFhh(-kx, -ky, rh);    % 用IEM的看看！！！！
% Fhhs = IEMFhh(-ksx, -ksy, rh);
% Ah = abs(0.5 * (css * Fhh + cs * Fhhs))^2;
% Bh = (css + cs) * real(fhh * (css * Fhh + cs * Fhhs));
% Ch = abs((css + cs) * fhh)^2;
% 
% Spv0 = Av / (Av + Bv + Cv);
% % if abs(Av + Bv + Cv) < 1e-5
% %     Spv0 = 1;
% % end
% Sph0 = Ah / (Ah + Bh + Ch);
% % if abs(Ah + Bh + Ch) < 1e-5
% %     Sph0 = 1;
% % end
% 
% % calc Spv & Sph (calc sigma_vv_c ... by IEM)
% % The constant 0.5*k^2*exp(-delta^2*(kz^2+ksz^2)) in sigma_* is ignored  
% % since  finally the ratio is what we need
% Ivvcn = 0.5 * ((nksz).^nv * Fvv + (nkz).^nv * Fvvs);
% Ivvn = (nksz + nkz).^nv * fvv * exp(-delta^2 * kz * ksz) + Ivvcn;
% % Spv0 = abs(Ivvcn(1)).^2 / abs(Ivvn(1)).^2;
% sigma_vv_c = real(sum(kdel2_n .* abs(Ivvcn).^2 .* Wn));
% sigma_vv_total = real(sum(kdel2_n .* abs(Ivvn).^2 .* Wn));
% 
% Ihhcn = 0.5 * ((nksz).^nv * Fhh + (nkz).^nv * Fhhs);
% Ihhn = (nksz + nkz).^nv * fhh * exp(-delta^2 * kz * ksz) + Ihhcn;
% % Sph0 = abs(Ihhcn(1)).^2 / abs(Ihhn(1)).^2;
% sigma_hh_c = real(sum(kdel2_n .* abs(Ihhcn).^2 .* Wn));
% sigma_hh_total = real(sum(kdel2_n .* abs(Ihhn).^2 .* Wn));
%--------------------------------------------------------------------------

% calc rvtran & rhtran
Spv = sigma_vv_c / sigma_vv_total;
Sph = sigma_hh_c / sigma_hh_total;
Gamv = 1.0 - Spv / Spv0;
Gamh = 1.0 - Sph / Sph0;

if (Gamv < 0.0)
    Gamv = 0.0;
end
if (Gamh < 0.0)
    Gamh = 0.0;
end

rvtran = rvi + (rvsp - rvi) * Gamv;
rhtran = rhi + (rhsp - rhi) * Gamh;
rvhtran = (rvtran - rhtran) / 2.0;

%% update the Kirchhoff coefficient by transition reflection function
rv = rvtran;
rh = rhtran;
rvh = rvhtran;
% 一直用入射角
% rv = rvi;
% rh = rhi;
% rvh = rvhi;

fvv = -((1-rv) * hsnv + eta_ist * (1 + rv) * vsnh) ...
    - (rh + rv) * vt * (hsnt + eta_ist * vsnd);
fhh = ((1 + rh) * vsnh + eta_ist * (1 - rh) * hsnv) ...
    - (rh + rv) * hd * (vsnd + eta_ist * hsnt);
fvh = (-(1 + rh) * hsnh + eta_ist * (1 - rh) * vsnv) ...
    + (rh + rv) * hd * (hsnd - eta_ist * vsnt);
fhv = ((1 - rv) * vsnv - eta_ist * (1 + rv) * hsnh) ...
    + (rh + rv) * vt * (vsnt - eta_ist * hsnd);

%% 计算单次散射<SqpSrs*>
% Kirchhoff coefficient
fqp_ = [fvv; fvh; fhv; fhh];

% Complementary coefficient
%   补偿场系数要一直使用入射角
rv = rvi;
rh = rhi;
rvh = rvhi;
u = -kx; v = -ky;
Fqp_ = [IEMFvv(u, v, rv); IEMFvh(u, v, rvh); IEMFhv(u, v, rvh); IEMFhh(u, v, rh)];
u = -ksx; v = -ksy;
Fqps_ = [IEMFvv(u, v, rv); IEMFvh(u, v, rvh); IEMFhv(u, v, rvh); IEMFhh(u, v, rh)];

Sqprs = zeros(4, 4);    % Single scattering term
% qprs (qp:i, rs:j)
for i = 1:4
    fqp = fqp_(i);
    Fqp = Fqp_(i);
    Fqps = Fqps_(i);
    for j = 1:4
        frs = fqp_(j);
        Frs = Fqp_(j);
        Frss = Fqps_(j);
        
        Iqpn = (nksz + nkz).^nv .* fqp .* exp(-delta^2 * ksz * kz) ...
            + 0.5 * (nksz.^nv * Fqp + nkz.^nv * Fqps);
        Irsn = (nksz + nkz).^nv .* frs .* exp(-delta^2 * ksz * kz) ...
            + 0.5 * (nksz.^nv * Frs + nkz.^nv * Frss);

        Iqprsn = Iqpn .* conj(Irsn);
        Sqprs(i, j) = sum(Iqprsn .* kdel2_n .* Wn);
    end
end

C0 = k^2 / (8 * pi) * exp(-delta^2 * (ksz^2 + kz^2));
Ms = qprs2Mueller(C0 .* Sqprs);

% IEM坐标系转换到FSA坐标系
U = diag([1, 1, -1, -1]);
Ms = U * Ms;

% Shadowing
if shadow_flag == true
    Shadow = R1_shadowing(theta_i, rss);
    Ms = Ms .* Shadow;
end

%% for debug

% Coeff_test = zeros(6, 1);
% sca_test = Gamv;    % 
end

%% Other routines
function [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)
if abs(real(ksz - d * qi))<0.0000000001
    n1_ = [0; 0; 1];
else
    Zx = -(ksx + u) / (ksz - d * qi);
    Zy = -(ksy + v) / (ksz - d * qi);
    n1_ = [-Zx; -Zy; 1];   % 局部法向矢量（但不是单位矢量）
end
if abs(real(kz + d * qi))<0.0000000001
    n2_ = [0; 0; 1];
else
    Zx1 = (kx + u) / (kz + d* qi);
    Zy1 = (ky + v) / (kz + d* qi);
    n2_ = [-Zx1; -Zy1; 1];
end

g_ = [u; v; -d*qi]; % 注意应该是-d*qi
end

function [C] = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)
% k1为入射方媒质的波数
C = zeros(6, 1);
C(1) = k1 * dot(hs_, cross(n1_, cross(n2_, h_)));
C(2) = dot(hs_, cross(n1_, cross(cross(n2_, v_), g_)));
C(3) = dot(hs_, cross(n1_, dot(n2_, v_) * g_));
C(4) = k1 * dot(vs_, cross(n1_, cross(n2_, v_)));
C(5) = dot(vs_, cross(n1_, cross(cross(n2_, h_), g_)));
C(6) = dot(vs_, cross(n1_, dot(n2_, h_) * g_));

% just for test
% C(1) = dot(hs_, cross(n1_, cross(n2_, h_))); % no k1
% C(4) = dot(vs_, cross(n1_, cross(n2_, v_))); % no k1
end

function [B] = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)
% k1为入射方媒质的波数
B = zeros(6, 1);
B(1) = k1 * dot(vs_, cross(n1_, cross(n2_, h_)));
B(2) = dot(vs_, cross(n1_, cross(cross(n2_, v_), g_)));
B(3) = dot(vs_, cross(n1_, dot(n2_, v_) * g_));
B(4) = k1 * dot(hs_, cross(n1_, cross(n2_, v_)));
B(5) = dot(hs_, cross(n1_, cross(cross(n2_, h_), g_)));
B(6) = dot(hs_, cross(n1_, dot(n2_, h_) * g_));

% just for test
% B(1) = dot(vs_, cross(n1_, cross(n2_, h_))); % no k1
% B(4) = dot(hs_, cross(n1_, cross(n2_, v_))); % no k1
end

function [M] = qprs2Mueller(Sqprs)
% Convert Sqprs matrix to Mueller matrix
%           =Sqprs(Sqp*conj(Srs))=
% vvvv  vvvh    vvhv    vvhh
% vhvv  vhvh    vhhv    vhhh
% hvvv  hvvh    hvhv    hvhh
% hhvv  hhvh    hhhv    hhhh
%
%           =M (Mueller matrix)=
% <Svvvv>       <Svhvh>         Re<Svvvh>           -Im<Svvvh>
% <Shvhv>       <Shhhh>         Re<Shvhh>           -Im<Shvhh>
% 2*Re<Svvhv>   2*Re<Svhhh>     Re<Svvhh+Svhhv>     -Im<Svvhh-Svhhv>
% 2*Im<Svvhv>   2*Im<Svhhh>     Im<Svvhh+Svhhv>     Re<Svvhh-Svhhv>
M = zeros(4, 4);
M(1, 1) = Sqprs(1, 1); M(1, 2) = Sqprs(2, 2); M(1, 3) = real(Sqprs(1, 2)); M(1, 4) = -imag(Sqprs(1, 2));
M(2, 1) = Sqprs(3, 3); M(2, 2) = Sqprs(4, 4); M(2, 3) = real(Sqprs(3, 4)); M(2, 4) = -imag(Sqprs(3, 4));
M(3, 1) = 2*real(Sqprs(1, 3)); M(3, 2) = 2*real(Sqprs(2, 4)); M(3, 3) = real(Sqprs(1, 4)+Sqprs(2, 3)); M(3, 4) = -imag(Sqprs(1, 4)-Sqprs(2, 3));
M(4, 1) = 2*imag(Sqprs(1, 3)); M(4, 2) = 2*imag(Sqprs(2, 4)); M(4, 3) = imag(Sqprs(1, 4)+Sqprs(2, 3)); M(4, 4) = real(Sqprs(1, 4)-Sqprs(2, 3));

M = real(M);
end

function [p] = R1_shadowing(theta, s)
% 单次散射Shadowing function R1 by Smith[1967]
% theta: 入射角 (rad)
% s: 均方根坡度 (rad)
% p:代表表面某个点不会被遮挡的概率
ct = cot(theta);
f = 0.5*(sqrt(2/pi) * s / ct * exp(-ct^2 / 2 / s^2) - erfc(ct/s/sqrt(2)));
p = (1 - 0.5*erfc(ct / s / sqrt(2))) / (1 + f);
end

function [p] = R2_shadowing(theta, s)
% 单次散射Shadowing function R2 by Smith[1967]
% theta: 入射角 (rad)
% s: 均方根坡度 (rad)
% p:代表局部坡度和入射波束垂直时该点不处于阴影区的条件概率
ct = cot(theta);
f = 0.5*(sqrt(2/pi) * s / ct * exp(-ct^2 / 2 / s^2) - erfc(ct/s/sqrt(2)));
p = 1 / (1 + f);
end

function [C] = Ccoeff2(k, cs, s, css, ss, csfs, sfs, kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)
    si = s;
    sis = ss;
    qslp = d * qi;
    q = d * qi;

    kxu = kx + u;
    ksxu = ksx + u;
    kyv = ky + v;
    ksyv = ksy + v;
    if(abs(real(ksz-qslp))<0.0000000001)
        zx = 0.0 ;
        zy = 0.0 ;
    else
        zx =(-ksxu)./(ksz-qslp);
        zy = -(ksyv)./(ksz-qslp);
    end
    if(abs(real(kz+qslp))<0.0000000001)
        zxp = 0.0 ;
        zyp = 0.0 ;
    else
        zxp =(kxu)./(kz+qslp);
        zyp =(kyv)./(kz+qslp);
    end

    C = zeros(6, 1);
    C(1) = -csfs.*(-1.0-zx.*zxp) + sfs.*zxp.*zy;
    C(2) = -csfs.*(-cs.*q-cs.*u.*zx-q.*si.*zxp-si.*u.*zx.*zxp-cs.*v.*zyp-si.*v.*zx.*zyp) ...
        + sfs.*(cs.*u.*zy+si.*u.*zxp.*zy+q.*si.*zyp-cs.*u.*zyp+si.*v.*zy.*zyp);
    C(3) = -csfs.*(si.*u-q.*si.*zx-cs.*u.*zxp+cs.*q.*zx.*zxp) ...
        + sfs.*(-si.*v+cs.*v.*zxp+q.*si.*zy-cs.*q.*zxp.*zy);
    C(4) = -css.*sfs.*(-si.*zyp+cs.*zx.*zyp) - csfs.*css.*(-cs-si.*zxp-cs.*zy.*zyp) ...
        + sis.*(-cs.*zx-si.*zx.*zxp-si.*zy.*zyp);
    C(5) = -css.*sfs.*(-v.*zx+v.*zxp) - csfs.*css.*(q+u.*zxp+v.*zy) ...
        + sis.*(q.*zx+u.*zx.*zxp+v.*zxp.*zy);
    C(6) = -css.*sfs.*(-u.*zyp+q.*zx.*zyp) - csfs.*css.*(v.*zyp-q.*zy.*zyp)...
        + sis.*(v.*zx.*zyp-u.*zy.*zyp);

    C(1) = C(1) * k;
    C(4) = C(4) * k;
end