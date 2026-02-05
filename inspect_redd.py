import tables
import numpy as np

print("Inspecting REDD H5 file...")

try:
    with tables.open_file('redd.h5', 'r') as h5file:
        print("Root group:", h5file.root)
        print("Root children:", [child for child in h5file.root])

        # Try to access building1
        if hasattr(h5file.root, 'building1'):
            building = h5file.root.building1
            print("Building1 groups:", [g for g in building])

            # Try to access elec group
            if hasattr(building, 'elec'):
                elec = building.elec
                print("Elec groups:", [e for e in elec])

                # Try to access meter1
                if hasattr(elec, 'meter1'):
                    meter1 = elec.meter1
                    print("Meter1 children:", [c for c in meter1])

                    # Try to access active data
                    if hasattr(meter1, 'active'):
                        active = meter1.active
                        print("Active shape:", active.shape)
                        if active.size > 5:
                            print("First 5 active values:", active[:5])
                    else:
                        print("No active data in meter1")

                    # Try to access name
                    if hasattr(meter1, 'name'):
                        name = meter1.name
                        print("Name:", name)
                    else:
                        print("No name in meter1")
                else:
                    print("No meter1 in elec")
            else:
                print("No elec in building1")
        else:
            print("No building1 group")

except Exception as e:
    print(f"Error inspecting REDD file: {str(e)}")
    import traceback
    traceback.print_exc()