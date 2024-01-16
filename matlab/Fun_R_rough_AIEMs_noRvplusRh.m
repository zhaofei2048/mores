function [Ms, Coeff_test, sca_test] = Fun_R_rough_AIEMs_noRvplusRh(f, angle_s, angle_i, epsilon_1, epsilon_2, sp, SpR, x, shadow_flag)
%FUN_R_ROUGH_AIEMs
%使用AIEM模型计算向上单次散射的Mueller矩阵（非相干分量,FSA坐标系）
%   来自Fun_R_rough_AIEMs，区别是在计算fvv和fhh时省略Rv+Rh项
%   Author: Fei Zhao
%   Create: 2022-11-03
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
%          Msk  : single scattering Kirchhoff term of the Mueller matrix
%          Mskc : single scattering Kirchhoff cross term of the Mueller matrix
%          Msc  : single scattering Complementary term of the Mueller matrix
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
%       Outline:
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
%
%   Logs:
%       2022-11-28: 关闭了针对后向散射以及散射角等于入射角时的数值稳定性扰动操作
%       2022-11-28: 尝试优化散射系数计算部分，避免重复计算
%       2022-12-6: v3 更改坡度计算，避免除0
%       2023-1-2：删除use_C2这个变量
%       2023-1-15: 去除wvnb计算时的abs
%       2023-2-10:
%       确保theta_i和theta_s始终不严格相等，同时发现粗糙尺度过大时，AIEM大入射角计算不正确，而在双战散射计算时不要让入射角处在布儒斯特角附近
%       2023-2-12: Fhv1中的B(6)项的负号纠正为正号
%                  为了数值稳定，改为永远不让入射角严格等于散射角，方式是发现相等时给散射角加上0.001度
%       2023-2-13: 过渡函数权重计算调整为始终使用后向散射时的计算方式（权重仅和入射角、介电常数、粗糙度&谱有关，特别注意wvnb不要引入散射角影响）
%                   添加shadow_flag参数控制是否添加阴影遮蔽函数
%       2023-2-18: 修正IEM坐标系到FSA坐标系的转换关系（原来写成了BSA到FSA的转换）

% test flag
% disp("test code 2521")

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
%     % disp('后向散射');
%     angle_s(1) = angle_s(1) + 0.0001; % 确保数值稳定
% end

% 下面的操作用以避免双站散射时散射角等于入射角时出现的不连续性！！！！！！！！！！！
%  引起不连续的原因还不知道！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% if angle_s(1) == angle_i(1)
%     angle_s(1) = angle_s(1) + 0.0001; % 确保数值稳定
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
wvnb = sqrt(Kx^2 + Ky^2);
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
% test_tmp = normt;
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

%% 计算补偿场系数函数F(+-)vv1, F(+-)vv2, F(+-)hh1, F(+-)hh2, F(+-)vh1, F(+-)vh2, F(+-)hv1, F(+-)hv2
    % F(+-)vv1(u,v): 媒质1中向上/向下（d=+1,-1）传播的补偿场系数
    function F = Fvv1(u, v, d, rv)
        % 缩放后的补偿场系数（*(ksz-dqi)*(kz+dqi)
        rvp = (1 + rv);
        rvm = (1 - rv);
        qi = sqrt(k1^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -(rvm / qi) * rvp * C(1) + rvm / qi * rvm * C(2) + rvm / qi * rvp * C(3) ...
            + eta_ist * (rvp / qi * rvm * C(4) + rvp / qi * rvp * C(5) + rvp / qi * rvm * C(6));
    end

    % F(+-)vv2(u,v): 媒质2中向上/向下（d=+1,-1）传播的补偿场系数
    function F = Fvv2(u, v, d, rv)
        rvp = (1 + rv);
        rvm = (1 - rv);
        qi = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);
        % just for test
%         [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, sqrt(k1^2 - u^2 - v^2));
%         g_ = [u; v; -d*qi];

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = (rvp * mur / qi) * rvp * C(1) - rvp / qi * rvm * C(2) - rvp / qi / er * rvp * C(3) ...
            + eta_ist * (-rvm *er / qi * rvm * C(4) - rvm / qi * rvp * C(5) - rvm / qi /mur * rvm * C(6));
    end

    % F(+-)hh1(u,v)
    function F = Fhh1(u, v, d, rh)
        rhp = (1 + rh);
        rhm = (1 - rh);
        qi = sqrt(k1^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -rhp / qi * rhm * C(4) - rhp / qi * rhp * C(5) - rhp / qi * rhm * C(6) ...
            + eta_ist * (rhm / qi * rhp * C(1) - rhm / qi * rhm * C(2) - rhm / qi* rhp * C(3));
    end

    % F(+-)hh2(u,v)
    function F = Fhh2(u, v, d, rh)
        rhp = (1 + rh);
        rhm = (1 - rh);
        qi = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);
        % just for test
%         [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, sqrt(k1^2 - u^2 - v^2));
%         g_ = [u; v; -d*qi];

        C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = rhm * mur / qi * rhm * C(4) + rhm / qi * rhp * C(5) + rhm / qi /er * rhm * C(6) ...
            + eta_ist * (-rhp * er / qi * rhp * C(1) + rhp / qi * rhm * C(2) + rhp / qi / mur * rhp * C(3));
    end

    % F(+-)vh1(u,v)
    function F = Fvh1(u, v, d, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k1^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = rm / qi * rp * B(4) + rm / qi * rm * B(5) + rm / qi * rp * B(6) ...
            + eta_ist * (rp / qi * rm * B(1) - rp / qi * rp * B(2) - rp / qi * rm * B(3));
    end

    % F(+-)vh2(u,v)
    function F = Fvh2(u, v, d, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -rp * mur / qi * rp * B(4) - rp / qi * rm * B(5) - rp / qi / er * rp * B(6) ...
            + eta_ist * (-rm * er / qi * rm * B(1) + rm / qi * rp * B(2) + rm / qi / mur * rm * B(3));
    end

    % F(+-)hv1(u,v)
    function F = Fhv1(u, v, d, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k1^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = rm / qi * rp * B(1) - rm / qi * rm * B(2) - rm / qi * rp * B(3) ...
            + eta_ist * (rp / qi * rm * B(4) + rp / qi * rp * B(5) + rp / qi * rm * B(6));
    end    

    % F(+-)hv2(u,v)
    function F = Fhv2(u, v, d, rvh)
        rp = 1 + rvh;
        rm = 1 - rvh;
        qi = sqrt(k2^2 - u^2 - v^2);
        [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi);

        B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_);

        F = -rp * mur / qi * rp * B(1) + rp / qi * rm * B(2) + rp / qi / er * rp * B(3) ...
            + eta_ist * (-rm * er / qi * rm * B(4) - rm / qi * rp * B(5) - rm / qi / mur * rm * B(6));
    end

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

%------------ transition by AIEM-------------------------------------------
% % 这个过渡函数还有问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% % calc Spv0 & Sph0 (by AIEM)
% q1 = sqrt(k1^2 - kx^2 - ky^2);
% q1s = sqrt(k1^2 - ksx^2 - ksy^2);
% q2 = sqrt(k2^2 - kx^2 - ky^2);
% q2s = sqrt(k2^2 - ksx^2 - ksy^2);
% nq1 = q1 / k;
% nq1s = q1s / k;
% nq2 = q2 / k;
% nq2s = q2s / k;
% 
% % rv = rvsp;
% % rh = rhsp;
% % rv = rv0;
% % rh = rh0;
% % rv = rvi;
% % rh = rhi;
% Ivvc0 = 0.25/k * (recp(nkz + nq1) * Fvv1(-kx, -ky, 1, rv) + recp(nksz - nq1s) * Fvv1(-ksx, -ksy, 1, rv)) ... % medium 1, upward
%         + 0.25/k * (recp(nkz - nq1) * Fvv1(-kx, -ky, -1, rv) + recp(nksz + nq1s) * Fvv1(-ksx, -ksy, -1, rv)) ... % medium 1, downward
%         + 0.25/k * (recp(nkz + nq2) * Fvv2(-kx, -ky, 1, rv) + recp(nksz - nq2s) * Fvv2(-ksx, -ksy, 1, rv)) ... % medium 2, upward
%         + 0.25/k * (recp(nkz - nq2) * Fvv2(-kx, -ky, -1, rv) + recp(nksz + nq2s) * Fvv2(-ksx, -ksy, -1, rv)); % medium 2, downward
% Ivv0 = (ksz + kz) * fvv + Ivvc0;
% Spv0 = abs(Ivvc0)^2 / abs(Ivv0)^2;
% 
% Ihhc0 = 0.25/k * (recp(nkz + nq1) * Fhh1(-kx, -ky, 1, rh) + recp(nksz - nq1s) * Fhh1(-ksx, -ksy, 1, rh)) ... % medium 1, upward
%         + 0.25/k * (recp(nkz - nq1) * Fhh1(-kx, -ky, -1, rh) + recp(nksz + nq1s) * Fhh1(-ksx, -ksy, -1, rh)) ... % medium 1, downward
%         + 0.25/k * (recp(nkz + nq2) * Fhh2(-kx, -ky, 1, rh) + recp(nksz - nq2s) * Fhh2(-ksx, -ksy, 1, rh)) ... % medium 2, upward
%         + 0.25/k * (recp(nkz - nq2) * Fhh2(-kx, -ky, -1, rh) + recp(nksz + nq2s) * Fhh2(-ksx, -ksy, -1, rh)); % medium 2, downward
% Ihh0 = (ksz + kz) * fhh + Ihhc0;
% Sph0 = abs(Ihhc0)^2 / abs(Ihh0)^2;
% 
% % calc Spv & Sph (by AIEM)
% Ivvcn = 0.25 * ((nksz - nq1).^(nv-1) * recp(nkz + nq1) * Fvv1(-kx, -ky, 1, rv) * exp(-delta^2*(q1^2 - q1 * ksz + q1 * kz)) + (nkz + nq1s).^(nv-1) * recp(nksz - nq1s) * Fvv1(-ksx, -ksy, 1, rv)) * exp(-delta^2*(q1s^2 - q1s * ksz + q1s * kz)) ... % medium 1, upward
%         + 0.25 * ((nksz + nq1).^(nv-1) * recp(nkz - nq1) * Fvv1(-kx, -ky, -1, rv) * exp(-delta^2*(q1^2 + q1 * ksz - q1 * kz)) + (nkz - nq1s).^(nv-1) * recp(nksz + nq1s) * Fvv1(-ksx, -ksy, -1, rv)) * exp(-delta^2*(q1s^2 + q1s * ksz - q1s * kz)) ... % medium 1, downward
%         + 0.25 * ((nksz - nq2).^(nv-1) * recp(nkz + nq2) * Fvv2(-kx, -ky, 1, rv) * exp(-delta^2*(q2^2 - q2 * ksz + q2 * kz)) + (nkz + nq2s).^(nv-1) * recp(nksz - nq2s) * Fvv2(-ksx, -ksy, 1, rv)) * exp(-delta^2*(q2s^2 - q2s * ksz + q2s * kz)) ... % medium 2, upward
%         + 0.25 * ((nksz + nq2).^(nv-1) * recp(nkz - nq2) * Fvv2(-kx, -ky, -1, rv) * exp(-delta^2*(q2^2 + q2 * ksz - q2 * kz)) + (nkz - nq2s).^(nv-1) * recp(nksz + nq2s) * Fvv2(-ksx, -ksy, -1, rv)) * exp(-delta^2*(q2s^2 + q2s * ksz - q2s * kz)); % medium 2, downward
% Ivvcn = Ivvcn / k^2;    % 因为这里的F系数多乘以了(ksz-dq)*(kz+dq)
% Ivvn = (nksz + nkz).^nv .* fvv .* exp(-delta^2 * ksz * kz) + Ivvcn;
% sigma_vv_c = real(sum(kdel2_n .* abs(Ivvcn).^2 .* Wn));
% sigma_vv_total = real(sum(kdel2_n .* abs(Ivvn).^2 .* Wn));
% 
% Ihhcn = 0.25 * ((nksz - nq1).^(nv-1) * recp(nkz + nq1) * Fhh1(-kx, -ky, 1, rh) * exp(-delta^2*(q1^2 - q1 * ksz + q1 * kz)) + (nkz + nq1s).^(nv-1) * recp(nksz - nq1s) * Fhh1(-ksx, -ksy, 1, rh)) * exp(-delta^2*(q1s^2 - q1s * ksz + q1s * kz)) ... % medium 1, upward
%         + 0.25 * ((nksz + nq1).^(nv-1) * recp(nkz - nq1) * Fhh1(-kx, -ky, -1, rh) * exp(-delta^2*(q1^2 + q1 * ksz - q1 * kz)) + (nkz - nq1s).^(nv-1) * recp(nksz + nq1s) * Fhh1(-ksx, -ksy, -1, rh)) * exp(-delta^2*(q1s^2 + q1s * ksz - q1s * kz)) ... % medium 1, downward
%         + 0.25 * ((nksz - nq2).^(nv-1) * recp(nkz + nq2) * Fhh2(-kx, -ky, 1, rh) * exp(-delta^2*(q2^2 - q2 * ksz + q2 * kz)) + (nkz + nq2s).^(nv-1) * recp(nksz - nq2s) * Fhh2(-ksx, -ksy, 1, rh)) * exp(-delta^2*(q2s^2 - q2s * ksz + q2s * kz)) ... % medium 2, upward
%         + 0.25 * ((nksz + nq2).^(nv-1) * recp(nkz - nq2) * Fhh2(-kx, -ky, -1, rh) * exp(-delta^2*(q2^2 + q2 * ksz - q2 * kz)) + (nkz - nq2s).^(nv-1) * recp(nksz + nq2s) * Fhh2(-ksx, -ksy, -1, rh)) * exp(-delta^2*(q2s^2 + q2s * ksz - q2s * kz)); % medium 2, downward
% Ihhcn = Ihhcn / k^2;
% Ihhn = (nksz + nkz).^nv .* fhh .* exp(-delta^2 * ksz * kz) + Ihhcn;
% sigma_hh_c = real(sum(kdel2_n .* abs(Ihhcn).^2 .* Wn));
% sigma_hh_total = real(sum(kdel2_n .* abs(Ihhn).^2 .* Wn));
%--------------------------------------------------------------------------

% calc rvtran & rhtran
Spv = sigma_vv_c / sigma_vv_total;
Sph = sigma_hh_c / sigma_hh_total;
Gamv = 1.0 - Spv / Spv0;
Gamh = 1.0 - Sph / Sph0;

if (Gamv < 0.0)
%     disp(['Phis_s: ', num2str(rad2deg(phi_s))]);
%     disp(['Gamv: ', num2str(Gamv)]);
    Gamv = 0.0;
end
if (Gamh < 0.0)
%     disp(['Phis_s: ', num2str(rad2deg(phi_s))]);
%     disp(['Gamh: ', num2str(Gamh)]);
    Gamh = 0.0;
end
rvtran = rvi + (rvsp - rvi) * Gamv;
rhtran = rhi + (rhsp - rhi) * Gamh;
rvhtran = (rvtran - rhtran) / 2.0;

%% update the Kirchhoff coefficient by transition reflection   
rv = rvtran;
rh = rhtran;
rvh = rvhtran;
% [rv, rh] = Fun_Fresnel_average(rad2deg(theta_i), er, rss, rss);
% 一直用入射角
% rv = rvi;
% rh = rhi;
% rvh = rvhi;

fvv = -((1-rv) * hsnv + eta_ist * (1 + rv) * vsnh);
fhh = ((1 + rh) * vsnh + eta_ist * (1 - rh) * hsnv);
fvh = (-(1 + rh) * hsnh + eta_ist * (1 - rh) * vsnv);
fhv = ((1 - rv) * vsnv - eta_ist * (1 + rv) * hsnh);

%% 计算单次散射<SqpSrs*>
% Kirchhoff coefficient
fqp_ = [fvv; fvh; fhv; fhh];

% Complementary coefficient
%   补偿场系数要一直使用入射角
rv = rvi;
rh = rhi;
rvh = rvhi;
u = -kx; v = -ky;
Fqp1up_ = [Fvv1(u, v, 1, rv); Fvh1(u, v, 1, rvh); Fhv1(u, v, 1, rvh); Fhh1(u, v, 1, rh)];
Fqp1down_ = [Fvv1(u, v, -1, rv); Fvh1(u, v, -1, rvh); Fhv1(u, v, -1, rvh); Fhh1(u, v, -1, rh)];
Fqp2up_ = [Fvv2(u, v, 1, rv); Fvh2(u, v, 1, rvh); Fhv2(u, v, 1, rvh); Fhh2(u, v, 1, rh)];
Fqp2down_ = [Fvv2(u, v, -1, rv); Fvh2(u, v, -1, rvh); Fhv2(u, v, -1, rvh); Fhh2(u, v, -1, rh)];
u = -ksx; v = -ksy;
Fqp1ups_ = [Fvv1(u, v, 1, rv); Fvh1(u, v, 1, rvh); Fhv1(u, v, 1, rvh); Fhh1(u, v, 1, rh)];
Fqp1downs_ = [Fvv1(u, v, -1, rv); Fvh1(u, v, -1, rvh); Fhv1(u, v, -1, rvh); Fhh1(u, v, -1, rh)];
Fqp2ups_ = [Fvv2(u, v, 1, rv); Fvh2(u, v, 1, rvh); Fhv2(u, v, 1, rvh); Fhh2(u, v, 1, rh)];
Fqp2downs_ = [Fvv2(u, v, -1, rv); Fvh2(u, v, -1, rvh); Fhv2(u, v, -1, rvh); Fhh2(u, v, -1, rh)];

k12 = [k1, k2];
d12 = [+1, -1];

Iqpnall = zeros(Nmax, 4);
for i = 1:4
    fqp = fqp_(i);
    Fqpall = [Fqp1up_(i), Fqp1down_(i);  Fqp2up_(i), Fqp2down_(i)];
    Fqpsall = [Fqp1ups_(i), Fqp1downs_(i);  Fqp2ups_(i), Fqp2downs_(i)];
    Iqpn = (nksz + nkz).^nv .* fqp .* exp(-delta^2 * ksz * kz);
    % (ii: medium 1/2, jj: up or downward +-)
    for ii = 1:2
        q = sqrt(k12(ii)^2 - kx^2 - ky^2);
        nq = q / k; % normalized
        qs = sqrt(k12(ii)^2 - ksx^2 - ksy^2);
        nqs = qs / k; % normalized
          % just for test
%         q = sqrt(k12(1)^2 - kx^2 - ky^2);
%         nq = q / k; % normalized
%         qs = sqrt(k12(1)^2 - ksx^2 - ksy^2);
%         nqs = qs / k; % normalized
        for jj = 1:2
            d = d12(jj);

%             recp_kzq = recp(nkz + d * nq);
%             recp_kszqs = recp(nksz - d * nqs);

%             Iqpn = Iqpn + 0.25 * (nksz - d * nq).^(nv-1) .* recp_kzq .* Fqpall(ii, jj) .* exp(-delta^2 * (q^2 - d * q * ksz + d * q * kz)) / k^2 ...  % u = -kx, v = -ky
%                 + 0.25 * (nkz + d * nqs).^(nv-1) .* recp_kszqs .* Fqpsall(ii, jj) .* exp(-delta^2 * (qs^2 - d * qs * ksz + d * qs * kz)) / k^2; % u = -ksx, v = -ksy
            
            Iqpn = Iqpn + 0.25 * (nksz - d * nq).^nv .* Fqpall(ii, jj) .* exp(-delta^2 * (q^2 - d * q * ksz + d * q * kz)) ...  % u = -kx, v = -ky
                + 0.25 * (nkz + d * nqs).^nv .* Fqpsall(ii, jj) .* exp(-delta^2 * (qs^2 - d * qs * ksz + d * qs * kz)); % u = -ksx, v = -ksy
        end
    end
    Iqpnall(:, i) = Iqpn;
end
Sqprs = zeros(4, 4);    % Single scattering term
% qprs (qp:i, rs:j)
for i = 1:4
    Iqpn = Iqpnall(:, i);
    for j = 1:4
        Irsn = Iqpnall(:, j);
        Iqprsn = Iqpn .* conj(Irsn);
        Sqprs(i, j) = sum(Iqprsn .* kdel2_n .* Wn);
    end
end

%% 计算Mueller矩阵
C0 = k^2 / (8 * pi) * exp(-delta^2 * (ksz^2 + kz^2));
Ms = qprs2Mueller(C0 .* Sqprs);

% IEM坐标系转换到FSA坐标系
U = diag([1, 1, -1, -1]);
Ms = U * Ms;

% Shadowing
if true == shadow_flag
    Shadow = R1_shadowing(theta_i, rss);
    Ms = Ms .* Shadow;
end

%% for debug
% FFvv = Fqp1up_(1) + Fqp1down_(1) + Fqp1ups_(1) + Fqp1downs_(1) + ...
%     Fqp2up_(1) + Fqp2down_(1) + Fqp2ups_(1) + Fqp2downs_(1);
% 
% FFhh = Fqp1up_(4) + Fqp1down_(4) + Fqp1ups_(4) + Fqp1downs_(4) + ...
%     Fqp2up_(4) + Fqp2down_(4) + Fqp2ups_(4) + Fqp2downs_(4);
% 
% Coeff_test = zeros(6, 1);
% sca_test = Av+ Bv + Cv;
% sca_test = Gamh;
% Coeff_test = zeros(6, 1);
% sca_test = Gamv;    % 为什么谷点位置不动呢？
end

%% Other routines
function r = recp(x)
% 避免产生数值误差的求倒，当x很小时，让倒数值为0
if abs(real(x)) < 1e-10
    r = 0 .* x;
else
    r = 1 ./ x;
end

end

% function [n1_, n2_, g_] = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)
% % 为了避免除0，将n1_和n2_分别都乘以ksz-d*qi和kz+d*qi
% Zx = -(ksx + u);
% Zy = -(ksy + v);
% n1_ = [-Zx; -Zy; (ksz - d * qi)];   % 局部法向矢量（但不是单位矢量）
% 
% Zx1 = (kx + u);
% Zy1 = (ky + v);
% n2_ = [-Zx1; -Zy1; (kz + d* qi)];
% 
% g_ = [u; v; -d*qi]; % 注意应该是-d*qi
% end

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
