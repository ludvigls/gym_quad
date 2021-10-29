import numpy as np


def princip(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi
def ssa(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def Rzyx(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])
    ])
def Tzyx(phi, theta, psi):
    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.vstack([
        np.hstack([1, sphi*tth, cphi*tth]), 
        np.hstack([0, cphi, -sphi]),
        np.hstack([0, sphi/cth, cphi/cth])])

def J(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta, psi)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([R, zero]),
        np.hstack([zero, T])])
def Rz(psi):
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi, -spsi, 0]),
        np.hstack([spsi, cpsi, -0]),
        np.hstack([0, 0, 1])
    ])
def S_skew(a):
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    return np.vstack([
        np.hstack([0, -a3, a2]),
        np.hstack([a3, 0, -a1]),
        np.hstack([-a2, a1, 0])])

def Rzyx_dpsi(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([-spsi*cth, -cpsi*cphi-spsi*sth*sphi, cpsi*sphi-spsi*cphi*sth]),
        np.hstack([cpsi*cth, -spsi*cphi+sphi*sth*cpsi, spsi*sphi+sth*cpsi*cphi]),
        np.hstack([0, 0, 0])
    ])

def to_homogeneous(x):
    return np.array([x[0], x[1], 1])

def to_cartesian(x):
    return np.array([x[0], x[1]])

def _H(r):
    I3 = np.identity(3)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([I3, np.transpose(S_skew(r))]),
        np.hstack([zero, I3])])


def move_to_CO(A_CG, r_g):
    H = _H(r_g)
    Ht = np.transpose(H)
    A_CO = Ht.dot(A_CG).dot(H)
    return A_CO
