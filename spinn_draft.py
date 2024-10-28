"""
SPINN built from scratch for deep notch

"""

import deepxde as dde
import numpy as np
import os
import jax
import jax.numpy as jnp

dde.config.set_default_autodiff("forward")

"""
For simplicity: 'the plate' refers to the actual geometry with the notch
                'the grid' refers to the new grid, ie the notch geometry mapped to the square grid
"""

# geometry of the grid (which is the input of the SPINN) - the grid has dimensions 1.2x1.2, the the plate has dimensions 1x1 (with the notch)
x_max = 1.2
y_max = 1.2
E = 210e3  # Young's modulus
nu = 0.3  # Poisson's ratio

lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
mu = E / (2 * (1 + nu))  # Lame's second parameter

pstress = 1.0

sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

# geometery for the input of the spinn, ie the grid
geom = dde.geometry.Rectangle([0, 0], [x_max, y_max])

# importing the plate coordinates
nx=120
ny=120
dir_path = os.path.dirname(os.path.realpath(__file__))
Xp = np.loadtxt(os.path.join(dir_path, f"deep_notch_geo_mapping_{ny}x{nx}.txt"))


X_map_points = Xp[:, 0].reshape((ny, nx)).T
Y_map_points = Xp[:, 1].reshape((ny, nx)).T


#map the points on the grid to the actual points on the plate, so x_grid --> x_plate
def coordMap(x, padding=1e-6):
    x_pos = x[0]/x_max*(nx-1)*(1-2*padding) + padding
    y_pos = x[1]/y_max*(ny-1)*(1-2*padding) + padding

    x_mapped = jax.scipy.ndimage.map_coordinates(X_map_points, [x_pos, y_pos], order=1, mode='nearest')
    y_mapped = jax.scipy.ndimage.map_coordinates(Y_map_points, [x_pos, y_pos], order=1, mode='nearest')

    return jnp.stack((x_mapped, y_mapped), axis=0)


# function is called in hardBC. Is it to "map the boundary conditions" so we're able to use the grid geometery definition
#     instead of the actual geometry with the notch? Was wondering how boundary conditions are handled using the grid geometry, since
#     we can't use the soft boundary condition functions on certain elements of the geometry (like the notch)
def tensMap(tens, x):
    J = jax.jacobian(coordMap)(x)
    return J @ tens


def jacobian(f, x, i, j):
    if dde.backend.backend_name == "jax":
        return dde.grad.jacobian(f, x, i=i, j=j)[0]  
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)

# pde function, analogous to regular pinns but the input is the grid coordinates, so they
# need to be mapped to the actual plate coordinates before calculating the strains/stresses/...
def pde(x, f):
    #this part is where the error originates. When running the code, it seems to first call this function for the boundary points (changing
    # the number of boundary points changes the dimensions of the printed array)
    # afterwards, it seems te call the function for the test points (changing the number of test points changes the dimensions of the printed array)
    # However, instead of an array of for example (100, 1) it seems to print a list of two arrays with seemingly random dimensions (ie [(19,1), (19,1)])
    # This causes mishandling in the meshgrid function. I haven't been able to figure out the exact origin of why test calls the pde function this way
    print("################## ENTER PDE #############")
    print("x", x)
    print("################## JNP ARRAY #############")
    x = x[0] if isinstance(x, list) else x
    x = jnp.array(x)
    print("x", x)
    print("x[:, 0]", x[:, 0])
    print("x[:, 1]", x[:, 1])
    x_meshgrid = jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")
    x_mesh = [x_.reshape(-1) for x_ in x_meshgrid]

    
    # x_mesh = [x_.reshape(-1) for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
    # replaced by above 2 lines to see if it fixes the error (it doesn't)
    # # takes grid coordinates and maps them to the actual plate coordinates
    x = stack(x_mesh, axis=-1)
    x = jax.vmap(coordMap)(x)

    E_xx = jacobian(f, x, i=0, j=0) 
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y 
    momentum_y = Sxy_x + Syy_y 

    if dde.backend.backend_name == "jax":
        f = f[0]  # f[1] is the function used by jax to compute the gradients

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


"""
For the boundary conditions, can I create a geometry with the actual notch in it and apply soft boundary conditions (ie similar to
the regular PINN boundary condition definitions)? Or should I work with the grid coordinates (the square plate)? And if I need to use
the grid coordinates, how do I define the boundary conditions for the notch?
"""
"""
rect = dde.geometry.Rectangle([0, 0], [1.0, 1.0])
notch_rect = dde.geometry.Rectangle([0, 0.4], [0.4, 0.6])
notch_circle = dde.geometry.Disk([0.4, 0.5], 0.1)
geom = dde.geometry.CSGDifference(rect, notch_rect)
geom = dde.geometry.CSGDifference(geom, notch_circle)
"""


"""
def boundary_notch_circle(x, on_boundary):
    return on_boundary and dde.utils.isclose((x[0]-0.4)**2 + (x[1]-0.5)**2, 0.1**2) and 0.4 < x[0] <= 0.5 and 0.4 < x[1] < 0.6

def boundary_notch_rect_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.4) and 0 <= x[0] <= 0.4

def boundary_notch_rect_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.6) and 0 <= x[0] <= 0.4



syy_boundary_notch_rect_top = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_notch_rect_top, component=3)

syy_boundary_notch_rect_bottom = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_notch_rect_bottom, component=3)


def traction_arc_xx(x, f, _):
    theta = jnp.atan2(x[:, 1:2]-0.5, x[:, 0:1]-0.4)
    sxx = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) + lmbd * (dde.grad.jacobian(f, x, i=1, j=1))
    sxy = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    valuexx = sxx * jnp.cos(theta) + sxy * jnp.sin(theta)

    return valuexx


def traction_arc_yy(x, f, _):
    theta = jnp.atan2(x[:, 1:2]-0.5, x[:, 0:1]-0.4)
    syy = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) + lmbd * (dde.grad.jacobian(f, x, i=0, j=0))
    sxy = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    valueyy = sxy * jnp.cos(theta) + syy * jnp.sin(theta)

    return valueyy

t_boundary_notch_circlexx = dde.icbc.OperatorBC(geom, traction_arc_xx, boundary_notch_circle)
t_boundary_notch_circleyy = dde.icbc.OperatorBC(geom, traction_arc_yy, boundary_notch_circle)

bcs_hard = [syy_boundary_notch_rect_top,
     syy_boundary_notch_rect_bottom, t_boundary_notch_circlexx,
     t_boundary_notch_circleyy]
"""
bcs_hard = []

# temporarilu left empty because it is also leading to errors
def HardBC(x, f):
#     if x.shape[0] != f.shape[0]:
#         x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
#         x = stack(x_mesh, axis=-1)

#     Ux = f[:, 0] * x[:, 1]
#     Uy = f[:, 1] * x[:, 1]

#     U_mapped = jax.vmap(tensMap)(stack((Ux, Uy), axis=1), x)

#     Sxx = f[:, 2] * x[:, 0] * (x[:, 0] - 1)
#     Syy = f[:, 3] * (x[:, 1] - 1) + 0.01
#     Sxy = f[:, 4]

#     S = jnp.stack((Sxx, Sxy, Sxy, Syy), axis=1).reshape(-1, 2, 2)
#     S_mapped = jax.vmap(tensMap)(S, x)

#     Syy_mapped = S_mapped[:,1,1] * (y_max - x[:, 1]) + pstress

#     S_mapped = jnp.stack((S_mapped[:,0,0],Syy_mapped,(S_mapped[:,0,1]+S_mapped[:,1,0])/2), axis=1)
#     return jnp.concatenate((U_mapped, S_mapped), axis=1)
    return f

# initializing the model and training
num_point = 100
num_test = 289
num_boundary = 100
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"

layers = [2, 64, 64, 64, 128, 5]
net = dde.nn.SPINN(layers, activation, initializer, 'mlp')


data = dde.data.PDE(
    geom,
    pde,
    bcs_hard,
    num_domain=num_point,
    num_boundary=num_boundary,
    solution=None,
    num_test=num_test,
    is_SPINN=True,
)

net.apply_output_transform(HardBC)

callbacks = []

model = dde.Model(data, net)
# model.compile(optimizer, lr=0.001, metrics=["l2 relative error"], loss_weights=[1,1,1e2,1e2,1e2])
model.compile(optimizer, lr=0.001, metrics=None, loss_weights=[1,1,1e2,1e2,1e2])
losshistory, train_state = model.train(iterations=5000, display_every=1000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)