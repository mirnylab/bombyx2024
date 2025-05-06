import openmm
import simtk.unit

def spherical_confinement_many(
    sim_object,
    r,#="density",  # radius... should be calculated based on number of chains you put in a given sphere
    k=5.0,  # How steep the walls are
    #density=0.3,  # target density, measured in particles
    # per cubic nanometer (bond size is 1 nm)
    cell_size=100,
    invert=False,
    particles=None,
    name="spherical_confinement",
):
    """Constrain particles to be within a sphere.
    With no parameters creates sphere with density .3
    Parameters
    ----------
    r : float or "density", optional
        Radius of confining sphere. If "density" requires density,
        or assumes density = .3
    k : float, optional
        Steepness of the confining potential, in kT/nm
    #density : float, optional, <1 (remove this?)
    #    Density for autodetection of confining radius.
    #    Density is calculated in particles per nm^3,
    #    i.e. at density 1 each sphere has a 1x1x1 cube.
    cell_size : int
        
    invert : bool
        If True, particles are not confinded, but *excluded* from the sphere.
    particles : list of int
        The list of particles affected by the force.
        If None, apply the force to all particles.
    """
    
    # assert r < 0.5*cell_size

    force = openmm.CustomExternalForce( 
        "step(invert_sign*(r-aa)) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t); "
        "r = sqrt((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2 + tt^2); "
        "x1 = x - L*floor(x/L); " 
        "y1 = y - L*floor(y/L); " 
        "z1 = z - L*floor(z/L); " 
    )
    force.name = name

    particles = range(sim_object.N) if particles is None else particles
    
    center = 3*[cell_size/2]
    
    for i in particles:
        force.addParticle(int(i), [])

    #if r == "density":
    #    r = (3 * sim_object.N / (4 * 3.141592 * density)) ** (1 / 3.0)

    if sim_object.verbose:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    force.addGlobalParameter("kb", k * sim_object.kT / simtk.unit.nanometer)
    force.addGlobalParameter("aa", (r - 1.0 / k) * simtk.unit.nanometer) 
    force.addGlobalParameter("t", (1.0 / k) * simtk.unit.nanometer / 10.0) # keeps calc from breaking if all the other values = 0
    force.addGlobalParameter("tt", 0.01 * simtk.unit.nanometer) # keeps calculation of r from breaking if all the other values = 0
    force.addGlobalParameter("invert_sign", (-1) if invert else 1)
    
    force.addGlobalParameter("L", cell_size * simtk.unit.nanometer)
    force.addGlobalParameter("x0", center[0] * simtk.unit.nanometer)
    force.addGlobalParameter("y0", center[1] * simtk.unit.nanometer)
    force.addGlobalParameter("z0", center[2] * simtk.unit.nanometer)

    # TODO: move 'r' elsewhere?..
    sim_object.sphericalConfinementRadius = r

    return force



#openmm.CustomExternalForce() reads bottom up (if you put each one on new line) and right to left on same line, semicolon separates expressions


# conformation = grow_cubic(chroms*L, (n_per_chain**(1/3)+2) 

