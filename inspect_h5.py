import tables
import numpy as np
from scipy.io import savemat
import os

print("Inspecting UK-DALE H5 file...")

with tables.open_file('ukdale.h5', 'r') as h5file:
    print("Root group:", h5file.root)
    if hasattr(h5file.root, 'building1'):
        building = h5file.root.building1
        print("Building1 groups:", [g for g in building])
        try:
            meter1 = getattr(h5file.root.building1.elec, 'meter1')
            print("Meter1 children:", [c for c in meter1])
            if hasattr(meter1, 'active'):
                active = meter1.active
                print("Active shape:", active.shape)
                if active.size > 5:
                    print("First 5 active values:", active[:5])
            if hasattr(meter1, 'index'):
                index = meter1.index
                print("Index shape:", index.shape)
                if index.size > 5:
                    print("First 5 index values:", index[:5])
        except AttributeError as e:
            print("Error accessing meter1:", e)
    else:
        print("No building1 group")