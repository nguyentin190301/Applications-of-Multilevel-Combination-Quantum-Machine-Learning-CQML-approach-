import numpy as np
from qml.representations import get_slatm_mbtypes
from qml.representations import generate_slatm
import glob

#Generate the training input matrix of SLATM representations for 10,000 geometries - Start here

SLATM_rep_full = []

file_format = "geometries/BChl_*.pdb.xyz"
for file_name in glob.iglob(file_format):
    xyz_file = np.genfromtxt(fname=file_name, skip_header=2, dtype='unicode')
    symbols = xyz_file[:,0]
    coordinates = (xyz_file[:,1:])
    coordinates = coordinates.astype(np.float)
    print("symbols = ", symbols)
    print("coordinates = ", coordinates)

    symbols_len = len(symbols)
    print("symbols_len = ", symbols_len)
    #Convert the array of element names to the array of corresponding nuclear charges
    nuclear_charges = np.zeros(symbols_len)
    for i in range(symbols_len):
        if symbols[i] == "Mg":
            nuclear_charges[i] = int(12)
        elif symbols[i] == "O":
            nuclear_charges[i] = int(8)
        elif symbols[i] == "N":
            nuclear_charges[i] = int(7)
        elif symbols[i] == "C":
            nuclear_charges[i] = int(6)
        elif symbols[i] == "H":
            nuclear_charges[i] = int(1)
    nuclear_charges = np.array(nuclear_charges)
    print("nuclear_charges = ", nuclear_charges)
    #Generate SLATM representation of one geometry
    mbtypes = get_slatm_mbtypes([nuclear_charges])
    SLATM_rep = generate_slatm(coordinates, nuclear_charges, mbtypes)
    print("SLATM_rep = ", SLATM_rep)
    print("SLATM_rep.shape = ", SLATM_rep.shape)
    SLATM_rep_full.append(SLATM_rep)

print("SLATM_rep_full 1 = ", SLATM_rep_full)

#Save SLATM representations of 10,000 geometries
save_SLATM_name = 'Results/SLATM_rep_full.npz'
np.savez_compressed(save_SLATM_name, SLATM_rep_full_svd=SLATM_rep_full)

#Generate the training input matrix of SLATM representations for 10,000 geometries - End here