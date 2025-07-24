import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj_w_units import CEA_Obj

# Define propellants
fuel = 'Ethanol'
oxidizer = 'N2O'

CEA = CEA_Obj(propName='', oxName=oxidizer, fuelName=fuel,
              pressure_units='Pa',
              cstar_units='m/s',
              temperature_units='K',
              sonic_velocity_units='m/s',
              enthalpy_units='J/kg',
              density_units='kg/m^3',
              specific_heat_units='J/kg-K')

# Estimate chamber pressure
T_amb = 25  # degC
P_amb = 101325  # Pa
P_tank_guess = PropsSI('P', 'T', T_amb + 273.15, 'Q', 0, oxidizer)
inj_dP_target = 0.5
P_c_target = P_tank_guess * (1 - inj_dP_target)

print("------ Pressures ------")
print(f"Initial N2O Tank Pressure: {P_tank_guess * 1e-6:.2f} MPa")
print(f"Target chamber pressure: {P_c_target * 1e-6:.2f} MPa")

# Parameter ranges
OF_list = np.linspace(2.0, 8.0, 40)
eps_list = np.linspace(1, 20, 40)

# Allocate result grid
Isp_grid = np.zeros((len(OF_list), len(eps_list)))
T_grid = np.zeros_like(Isp_grid)

# Evaluate grid
for i, OF in enumerate(OF_list):
    for j, eps in enumerate(eps_list):
        try:
            isp = CEA.estimate_Ambient_Isp(Pc=P_c_target, MR=OF, eps=eps, Pamb=101325, frozen=0)[0]
            Tc = CEA.get_Tcomb(Pc=P_c_target, MR=OF)
            Isp_grid[i, j] = isp
            T_grid[i, j] = Tc
        except:
            Isp_grid[i, j] = np.nan
            T_grid[i, j] = np.nan

# Find optimal point
i_max, j_max = np.unravel_index(np.nanargmax(Isp_grid), Isp_grid.shape)
OF_opt = OF_list[i_max]
eps_opt = eps_list[j_max]
Isp_opt = Isp_grid[i_max, j_max]
Tc_opt = T_grid[i_max, j_max]

print("------ Optimal Sea-Level Performance ------")
print(f"Optimal OF ratio: {OF_opt:.2f}")
print(f"Optimal expansion ratio: {eps_opt:.1f}")
print(f"Sea-level Isp: {Isp_opt:.1f} s")
print(f"Combustion Temperature: {Tc_opt:.1f} K")

# Plot
OF_mesh, eps_mesh = np.meshgrid(eps_list, OF_list)
plt.figure(figsize=(8, 6))
cp = plt.contourf(eps_mesh, OF_mesh, Isp_grid, levels=30, cmap='viridis')
plt.colorbar(cp, label='Sea-Level Isp (s)')
plt.xlabel('Expansion Ratio ($\epsilon$)')
plt.ylabel('O/F Ratio')
plt.title('Sea-Level Isp for N2O / Ethanol\nPc = {:.2f} MPa'.format(P_c_target / 1e6))
plt.plot(eps_opt, OF_opt, 'rx', markersize=10, label='Optimal Point')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

