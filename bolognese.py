import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


Data = pd.read_pickle("Sim_Data_WithNoise.pkl")
#y = np.transpose(np.vstack((Data["x_tag"].to_numpy(),Data["y_tag"].to_numpy(),Data["z_tag"].to_numpy())))
y = Data["y_tag"].to_numpy()
phases = Data["Phases"].to_numpy()
powers = Data["Powers"].to_numpy()
original_pos = Data["robot_pos"]
spacing = 0.004
num_of_ys = len(phases)
data_list = []

max_x = max(original_pos[0][:,0]) #MERIKA EIXANE 1 PARAPANO KAI MERIKA 1 LIGOTERO

for i in range(num_of_ys):
    phase = phases[i]
    power = powers[i]
    org_pos = original_pos[i]
    org_x = org_pos[:,0]

    x_new = np.arange(0,max_x,spacing) # STATHERO MEGETHOS SE OLA TA DATA
    linear_interp_phase = interp1d(org_x,phase,kind='linear',fill_value='extrapolate')
    linear_interp_power = interp1d(org_x,power,kind='linear',fill_value='extrapolate')
    phase_new = linear_interp_phase(x_new)
    power_new = linear_interp_power(x_new)
    data_list.append({
        "x_tag": Data['x_tag'][i],
        "y_tag": Data['y_tag'][i],
        "z_tag": Data['z_tag'][i],
        "Antenna": Data['Antenna'][i],
        "robot_pos" : np.array([x_new, np.repeat(org_pos[0,1],len(x_new)), np.repeat(org_pos[0,2],len(x_new))]).T, # AYTO THA DOULEPSEI MONO AN TO Y KAI Z EINAI STATHERA
        "Phases": np.array(phase_new),
        "Powers": np.array(power_new)
    })

df = pd.DataFrame(data_list)


df.to_pickle("SimulationDataWithNoise.pkl")



