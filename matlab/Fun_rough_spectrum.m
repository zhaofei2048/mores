function [Wn, rss] = Fun_rough_spectrum(type, N, delta, L, wvnb, x )
%FUN_ROUGH_SPECTRUM Generate N-order spectrum of rough surface
%
%   params:
%       type <char>vector : the spectrum of the rough surface, i.e., the Fourier transform of corr function.
%       N int the order of the spectrum
%       delta int (meter) : RMS height
%       L int (meter) : correlation length of rough surface
%       wvnb double : K (W^n(K), W^n(K1, K2) where K = sqrt(K1^2+K2^2))
%       x scalar : coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correl. fnc.
%       
%       the type can be one of {'gauss', 'exp', 'x-power', 'x-exp', 'gauss-1d'}
%   return:
%       Wn double[N*1]
%       rss double (rad) : root mean square slope
%
%   Logs:
%       2022-11-29 更新了均方根坡度的计算
Norder = reshape(1:N, [N, 1]);  % shape(Wn) ~ (N, 1)
switch lower(type)
    case 'gauss'    % gaussian correlation function
        Wn = L^2 ./ (2 * Norder) .* exp(-(wvnb*L)^2 ./ (4 * Norder));
        rss = sqrt(2) * delta / L;
    case 'exp'      % exponential correlation function
        Wn = L^2 ./ Norder.^2 .* (1 + (wvnb*L)^2 ./ Norder.^2).^(-1.5);
        rss = delta / L;
    case 'x-power'  % x-power correlation function
        if wvnb == 0
            Wn = L^2 ./ (3 * Norder - 2);
        else
%             Wn = L^2 * (wvnb * L).^(-1 + x * Norder) .* besselk(1-x*Norder, wvnb*L);
            Wn = L^2 * (wvnb * L).^(-1 + x * Norder) .* besselk(1-x*Norder, wvnb*L)...
                ./ (2.^(x * Norder - 1) .* gamma(x * Norder));
        end
        rss = sqrt(x * 2) * delta / L;
    case 'x-exp'    % x-exponential correlation function
        Wn = zeros(N, 1);
        for n = 1:N
            tmp = integral(@(z)x_exponential_spectrum(z, wvnb, L, n, x), 0, 9);
            Wn(n) = L^2 / n^(2/x) * tmp;
        end
        rss = sqrt(4) * delta ./ L;
    case 'gauss-1d' % one dimensional gaussian correlation function (for 2-dimensional rough surface scattering)
        Wn = sqrt(pi ./ Norder) .* L .*  exp(-(wvnb*L)^2 ./ (4 * Norder));
        rss = sqrt(2) * delta / L;   % unknown
    otherwise
        error(['No matching surface spectrum for: ', type, 'use one of {''gauss'', ''exp'', ''x-power'', ''x-exp'', ...} instead']);
end

assert(size(Wn, 2)==1);
end

function [tmp] = x_exponential_spectrum(z,wvnb,L,n,xx)
tmp = exp(-abs(z).^xx) .* besselj(0, z.*wvnb.*L./(n.^(1/xx))).*z;
end
