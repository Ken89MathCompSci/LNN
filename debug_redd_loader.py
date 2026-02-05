import sys
sys.path.append('Source Code')

import tables
import warnings

def debug_explore_meters():
    """Debug the explore_available_meters function"""
    file_path = "redd.h5"

    try:
        with tables.open_file(file_path, 'r') as h5file:
            print("Root groups:", list(h5file.root))

            # Iterate through buildings
            for building_node in h5file.root:
                building_name = building_node._v_name
                print(f"Processing building: {building_name}")
                if building_name.startswith('building'):
                    building_number = int(building_name.replace('building', ''))
                    building_group = getattr(h5file.root, building_name)
                    print(f"Building {building_number} group: {building_group}")

                    # Check if electricity group exists
                    if hasattr(building_group, 'elec'):
                        elec_group = building_group.elec
                        print(f"Elec group: {elec_group}")

                        # Try to iterate through meters
                        print(f"Elec group items: {list(elec_group)}")

                        # Iterate through meters
                        for meter_node in elec_group:
                            meter_name = meter_node._v_name
                            print(f"Processing meter: {meter_name}")
                            if meter_name.startswith('meter'):
                                meter_number = int(meter_name.replace('meter', ''))
                                meter_group = getattr(elec_group, meter_name)
                                print(f"Meter {meter_number} group: {meter_group}")

                                # Get meter name if available
                                meter_display_name = f"Meter_{meter_number}"
                                if hasattr(meter_group, 'name'):
                                    meter_display_name = str(meter_group.name)
                                print(f"Meter display name: {meter_display_name}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_explore_meters()