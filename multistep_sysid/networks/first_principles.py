from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class AddedMass4DOFConfig(BaseModel):
    xud: float
    yvd: float
    ypd: float
    yrd: float
    kvd: float
    kpd: float
    krd: float
    nvd: float
    npd: float
    nrd: float


class MinimalManeuveringConfig(AddedMass4DOFConfig):
    m: float
    Ixx: float
    Izz: float
    xg: float
    zg: float
    rho_water: float
    disp: float
    gm: float
    g: float


class PropulsionManeuveringConfig(MinimalManeuveringConfig):
    wake_factor: float
    diameter: float
    Kt: Tuple[float, float, float]
    lx: float
    ly: float
    lz: float


class BasicPelicanMotionConfig(BaseModel):
    m: float
    g: float
    kt: float
    Ix: float
    Iy: float
    Iz: float
    kr: float


class MinimalManeuveringEquations(nn.Module):
    def __init__(self, time_delta: float, config: MinimalManeuveringConfig):
        super().__init__()

        self.dt = time_delta

        mass = config.m
        inertia = np.array([
            [config.Ixx, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, config.Izz]
        ])
        cog = np.array([
            config.xg, 0.0, config.zg
        ])
        m_rb = build_rigid_body_matrix(
            dof=4, mass=mass, inertia_matrix=inertia, center_of_gravity=cog
        )
        m_a = build_4dof_added_mass_matrix(config)
        # transpose for correct multiplication with batch
        self.inv_mass = nn.Parameter(torch.inverse(m_rb + m_a).t())
        self.inv_mass.requires_grad = False

        self.coriolis_rb = RigidBodyCoriolis4DOF(mass=mass, cog=cog)
        self.buoyancy = Buoyancy(
            rho_water=config.rho_water, displacement=config.disp, metacentric_height=config.gm,
            grav_acc=config.g
        )

    def forward(self, control, state):
        # control is not used but remains for a consistent interface between all whitebox models
        velocity = state[:, :4]
        position = torch.zeros((state.shape[0], 4), device=state.device, dtype=state.dtype)
        position[:, 2] = state[:, 4]

        tau_crb = self.coriolis_rb.forward(velocity)
        tau_hs = self.buoyancy.forward(position)
        tau_total = -tau_crb - tau_hs

        acceleration = torch.mm(tau_total, self.inv_mass)
        acceleration = torch.cat((
            acceleration,
            (position[:, 2] + self.dt * acceleration[:, 2]).unsqueeze(1)
        ), dim=1)
        return acceleration


class PropulsionManeuveringEquations(nn.Module):
    def __init__(self, time_delta: float, config: PropulsionManeuveringConfig):
        super().__init__()

        self.dt = time_delta

        mass = config.m
        inertia = np.array([
            [config.Ixx, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, config.Izz]
        ])
        cog = np.array([
            config.xg, 0.0, config.zg
        ])
        m_rb = build_rigid_body_matrix(
            dof=4, mass=mass, inertia_matrix=inertia, center_of_gravity=cog
        )
        m_a = build_4dof_added_mass_matrix(config)
        # transpose for correct multiplication with batch
        self.inv_mass = nn.Parameter(torch.inverse(m_rb + m_a).t())
        self.inv_mass.requires_grad = False

        self.coriolis_rb = RigidBodyCoriolis4DOF(mass=mass, cog=cog)
        self.buoyancy = Buoyancy(
            rho_water=config.rho_water,
            displacement=config.disp,
            metacentric_height=config.gm,
            grav_acc=config.g
        )
        self.propulsion = SymmetricRudderPropellerPair(
            wake_factor=config.wake_factor,
            diameter=config.diameter,
            rho_water=config.rho_water,
            thrust_coefficient=np.array(config.Kt),
            propeller_location=np.array([config.lx, config.ly, config.lz])
        )

    def forward(self, control, state):
        velocity = state[:, :4]
        position = torch.zeros((state.shape[0], 4), device=state.device, dtype=state.dtype)
        position[:, 2] = state[:, 4]

        tau_crb = self.coriolis_rb.forward(velocity)
        tau_hs = self.buoyancy.forward(position)
        tau_propulsion = self.propulsion.forward(control, velocity)
        tau_total = tau_propulsion - tau_crb - tau_hs

        acceleration = torch.mm(tau_total, self.inv_mass)
        acceleration = torch.cat((
            acceleration,
            (position[:, 2] + self.dt * acceleration[:, 2]).unsqueeze(1)
        ), dim=1)
        return acceleration


def build_rigid_body_matrix(dof, mass, inertia_matrix, center_of_gravity):
    if dof not in {3, 4, 6}:
        raise ValueError('Only degrees of freedom are 3, 4, and 6')
    if inertia_matrix.shape != (3, 3):
        raise ValueError('Expected shape of inertia matrix is (3,3). If values are unused due to DOF, set'
                         'to zero, e.g. I_yy for 4-DOF.')
    if center_of_gravity.size != 3:
        raise ValueError('Center of gravity is expected to have 3 coordinates. Set zg=0 for 3-DOF.')

    m = mass
    ixx = inertia_matrix[0, 0]
    ixy = inertia_matrix[0, 1]
    ixz = inertia_matrix[0, 2]
    iyx = inertia_matrix[1, 0]
    iyy = inertia_matrix[1, 1]
    iyz = inertia_matrix[1, 2]
    izx = inertia_matrix[2, 0]
    izy = inertia_matrix[2, 1]
    izz = inertia_matrix[2, 2]
    xg = center_of_gravity[0]
    yg = center_of_gravity[1]
    zg = center_of_gravity[2]

    mrb = torch.tensor([
        [m, 0.0, 0.0, 0.0, m * zg, -m * yg],
        [0.0, m, 0.0, -m * zg, 0.0, m * xg],
        [0.0, 0.0, m, m * yg, -m * xg, 0.0],
        [0.0, -m * zg, m * yg, ixx, -ixy, -ixz],
        [m * zg, 0, -m * xg, -iyx, iyy, -iyz],
        [-m * yg, m * xg, 0.0, -izx, -izy, izz]
    ]).float()

    if dof == 6:
        mrb = mrb
    elif dof == 4:
        mask = (0, 1, 3, 5)  # surge, sway, roll, yaw
        mrb = mrb[mask, :][:, mask]
    else:
        mask = (0, 1, 5)  # surge, sway, yaw
        mrb = mrb[mask, :][:, mask]

    return mrb


def build_4dof_added_mass_matrix(config: AddedMass4DOFConfig):
    xud = config.xud
    yvd = config.yvd
    ypd = config.ypd
    yrd = config.yrd
    kvd = config.kvd
    kpd = config.kpd
    krd = config.krd
    nvd = config.nvd
    npd = config.npd
    nrd = config.nrd

    ma = torch.tensor([
        [xud, 0.0, 0.0, 0.0],
        [0.0, yvd, ypd, yrd],
        [0.0, kvd, kpd, krd],
        [0.0, nvd, npd, nrd]
    ]).float()
    ma = -ma
    return ma


class RigidBodyCoriolis4DOF(nn.Module):
    def __init__(self, mass, cog):
        super().__init__()

        if cog.size != 3:
            raise ValueError('Center of gravity is expected as (x_g, y_g, z_g). Assume y_g=0.')

        m = mass
        xg = cog[0]
        zg = cog[2]

        # transpose for correct multiplication with batch
        self.crb = nn.Parameter(torch.tensor([
            [0.0, -m, m*zg, -m*xg],
            [m, 0.0, 0.0, 0.0],
            [-m*zg, 0.0, 0.0, 0.0],
            [m*xg, 0.0, 0.0, 0.0]
        ]).t())
        self.crb.requires_grad = False

    def forward(self, velocity):
        r = velocity[:, 3].unsqueeze(1)
        tau_crb = torch.mm(r*velocity, self.crb)
        return tau_crb


class Buoyancy(nn.Module):
    def __init__(self, rho_water, grav_acc, displacement, metacentric_height):
        super().__init__()

        self.G = nn.Parameter(torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, rho_water*grav_acc*displacement*metacentric_height, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]))
        self.G.requires_grad = False

    def forward(self, position):
        tau_hs = torch.mm(position, self.G)

        return tau_hs


class SymmetricRudderPropellerPair(nn.Module):
    def __init__(self, wake_factor, diameter, rho_water, thrust_coefficient, propeller_location):
        super().__init__()

        if propeller_location.size != 3:
            raise ValueError('Propeller location relative to COG has expected form (lx, ly, lz).')

        self.w = nn.Parameter(torch.tensor(wake_factor).float())
        self.w.requires_grad = False
        self.d = nn.Parameter(torch.tensor(diameter).float())
        self.d.requires_grad = False
        self.rho = nn.Parameter(torch.tensor(rho_water).float())
        self.rho.requires_grad = False
        self.kt = nn.Parameter(torch.tensor(thrust_coefficient).float())
        self.kt.requires_grad = False
        self.lx = nn.Parameter(torch.tensor(propeller_location[0]).float())
        self.lx.requires_grad = False
        self.ly = nn.Parameter(torch.tensor(propeller_location[1]).float())
        self.ly.requires_grad = False
        self.lz = nn.Parameter(torch.tensor(propeller_location[2]).float())
        self.lz.requires_grad = False

    def forward(self, control, velocity):
        """
        :param control: (n [1/min], delta_left [rad], delta_right [rad])
        :param velocity:
        :return: control forces
        """
        u = velocity[:, 0]
        v = velocity[:, 1]
        nl = (1.0/60.0) * control[:, 0]
        nr = (1.0/60.0) * control[:, 0]
        deltal = control[:, 1]
        deltar = control[:, 2]

        val = F.relu((1.0-self.w)*(torch.cos(deltal)*u + torch.sin(deltal)*v))
        var = F.relu((1.0-self.w)*(torch.cos(deltar)*u + torch.sin(deltar)*v))

        xl = torch.cat((
            (self.rho*self.d**2*val*val).unsqueeze(1),
            (self.rho*self.d**3*val*nl).unsqueeze(1),
            (self.rho*self.d**4*nl*nl).unsqueeze(1)
        ), dim=1)
        tl = F.relu(xl @ self.kt)

        xr = torch.cat((
            (self.rho*self.d**2*var*var).unsqueeze(1),
            (self.rho*self.d**3*var*nr).unsqueeze(1),
            (self.rho*self.d**4*nr*nr).unsqueeze(1)
        ), dim=1)
        tr = F.relu(xr @ self.kt)

        fxl = torch.cos(deltal) * tl
        fxr = torch.cos(deltar) * tr
        fyl = torch.sin(deltal) * tl
        fyr = torch.sin(deltar) * tr

        tau_control = torch.cat((
            (fxl + fxr).unsqueeze(1),
            (fyl + fyr).unsqueeze(1),
            (self.lz * (-fyl - fyr)).unsqueeze(1),
            (self.lx * (fyl + fyr) + self.ly * (fxl - fxr)).unsqueeze(1)
        ), dim=1)

        return tau_control


class BasicPelicanMotionEquations(nn.Module):
    """
    Motion model of quadrotor without propulsion.
    """
    def __init__(self, time_delta: float, config: BasicPelicanMotionConfig):
        super().__init__()

        self.mass = config.m
        self.gravity = config.g
        self.kt = config.kt

        self.ix = config.Ix
        self.iy = config.Iy
        self.iz = config.Iz
        self.kr = config.kr

        self.dt = time_delta

    def forward(self, control, state):
        phi = state[:, 0]
        theta = state[:, 1]

        xdot = state[:, 2]
        ydot = state[:, 3]
        zdot = state[:, 4]

        p = state[:, 5]
        q = state[:, 6]
        r = state[:, 7]

        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        ttheta = torch.tan(theta)

        # linear accelerations
        xdotdot = -self.kt * xdot / self.mass
        ydotdot = -self.kt * ydot / self.mass
        zdotdot = -self.kt * zdot / self.mass - self.gravity

        # angular accelerations
        pdot = -((self.iz - self.iy) / self.ix) * q * r - (self.kr * p / self.ix)
        qdot = -((self.ix - self.iz) / self.iy) * p * r - (self.kr * q / self.iy)
        rdot = -((self.iy - self.iz) / self.iz) * p * q - (self.kr * r / self.iz)

        # angular velocities
        phidot = p + q * sphi * ttheta + r * cphi * ttheta
        thetadot = q * cphi - r * sphi

        acceleration = torch.cat((
            phidot.unsqueeze(1),
            thetadot.unsqueeze(1),
            xdotdot.unsqueeze(1),
            ydotdot.unsqueeze(1),
            zdotdot.unsqueeze(1),
            pdot.unsqueeze(1),
            qdot.unsqueeze(1),
            rdot.unsqueeze(1)
        ), dim=1)

        return acceleration
