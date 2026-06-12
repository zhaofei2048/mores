"""Common functions

Author: Fei Zhao
Create: 2024-01-16
"""
import numpy as np
from scipy.special import erfc


#==============================================
# For IEM-based models
#==============================================
def qprs2Mueller(Sqprs):
    """Convert Sqprs matrix to Mueller matrix

    Args:
        =Sqprs(Sqp*conj(Srs))=
                 vvvv  vvvh    vvhv    vvhh
                 vhvv  vhvh    vhhv    vhhh
                 hvvv  hvvh    hvhv    hvhh
                 hhvv  hhvh    hhhv    hhhh
    
    Returns:
         =M (Mueller matrix)=
             <Svvvv>       <Svhvh>         Re<Svvvh>           -Im<Svvvh>
             <Shvhv>       <Shhhh>         Re<Shvhh>           -Im<Shvhh>
             2*Re<Svvhv>   2*Re<Svhhh>     Re<Svvhh+Svhhv>     -Im<Svvhh-Svhhv>
             2*Im<Svvhv>   2*Im<Svhhh>     Im<Svvhh+Svhhv>     Re<Svvhh-Svhhv>
    """
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = Sqprs[0, 0]; M[0, 1] = Sqprs[1, 1]; M[0, 2] = np.real(Sqprs[0, 1]); M[0, 3] = -np.imag(Sqprs[0, 1])
    M[1, 0] = Sqprs[2, 2]; M[1, 1] = Sqprs[3, 3]; M[1, 2] = np.real(Sqprs[2, 3]); M[1, 3] = -np.imag(Sqprs[2, 3])
    M[2, 0] = 2*np.real(Sqprs[0, 2]); M[2, 1] = 2*np.real(Sqprs[1, 3]); M[2, 2] = np.real(Sqprs[0, 3]+Sqprs[1, 2]); M[2, 3] = -np.imag(Sqprs[0, 3]-Sqprs[1, 2])
    M[3, 0] = 2*np.imag(Sqprs[0, 2]); M[3, 1] = 2*np.imag(Sqprs[1, 3]); M[3, 2] = np.imag(Sqprs[0, 3]+Sqprs[1, 2]); M[3, 3] = np.real(Sqprs[0, 3]-Sqprs[1, 2])

    M = np.real(M)

    return M


def R1_shadowing(theta, s):
    """单次散射Shadowing function R1 by Smith[1967]
    
    Args:
        theta: incidence angle (rad)
        s: RMS slope (rad)
    
    Returns:
        p: 代表表面某点不会被遮挡的概率
    """
    ct = 1 / np.tan(theta)
    f = 0.5*(np.sqrt(2/np.pi) * s / ct * np.exp(-ct**2 / 2 / s**2) - erfc(ct/s/np.sqrt(2)))
    p = (1 - 0.5*erfc(ct / s / np.sqrt(2))) / (1 + f)

    return p


def R2_shadowing(theta, s):
    """单次散射Shadowing function R2 by Smith[1967]
    
    Args:
        theta: incidence angle (rad)
        s: RMS slope (rad)
    
    Returns:
        p: 代表表面某点不会被遮挡的概率
    """
    ct = 1 / np.tan(theta)
    f = 0.5*(np.sqrt(2/np.pi) * s / ct * np.exp(-ct**2 / 2 / s**2) - erfc(ct/s/np.sqrt(2)))
    p = 1 / (1 + f)

    return p


def Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_):
    """Calc the C coefficients for IEM-based models"""
    C = np.zeros(6, dtype=complex)
    C[0] = k1 * np.dot(hs_, np.cross(n1_, np.cross(n2_, h_)))
    C[1]  = np.dot(hs_, np.cross(n1_, np.cross(np.cross(n2_, v_), g_)))
    C[2]  = np.dot(hs_, np.cross(n1_, np.dot(n2_, v_) * g_))
    C[3]  = k1 * np.dot(vs_, np.cross(n1_, np.cross(n2_, v_)))
    C[4]  = np.dot(vs_, np.cross(n1_, np.cross(np.cross(n2_, h_), g_)))
    C[5]  = np.dot(vs_, np.cross(n1_, np.dot(n2_, h_) * g_))

    return C


def Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_):
    """Calc the B coefficients for IEM-based models"""
    B = np.zeros(6, dtype=complex)
    B[0] = k1 * np.dot(vs_, np.cross(n1_, np.cross(n2_, h_)))
    B[1] = np.dot(vs_, np.cross(n1_, np.cross(np.cross(n2_, v_), g_)))
    B[2] = np.dot(vs_, np.cross(n1_, np.dot(n2_, v_) * g_))
    B[3] = k1 * np.dot(hs_, np.cross(n1_, np.cross(n2_, v_)))
    B[4] = np.dot(hs_, np.cross(n1_, np.cross(np.cross(n2_, h_), g_)))
    B[5] = np.dot(hs_, np.cross(n1_, np.dot(n2_, h_) * g_))

    return B


def calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi):
    if np.abs(np.real(ksz - d * qi))<0.0000000001:
        n1_ = np.array([0, 0, 1])
    else:
        Zx = -(ksx + u) / (ksz - d * qi)
        Zy = -(ksy + v) / (ksz - d * qi)
        n1_ = np.array([-Zx, -Zy, 1])   # 局部法向矢量（但不是单位矢量）
    
    if np.abs(np.real(kz + d * qi))<0.0000000001:
        n2_ = np.array([0, 0, 1])
    else:
        Zx1 = (kx + u) / (kz + d* qi)
        Zy1 = (ky + v) / (kz + d* qi)
        n2_ = np.array([-Zx1, -Zy1, 1])

    g_ = np.array([u, v, -d*qi]) # 注意应该是-d*qi

    return n1_, n2_, g_