import sys
sys.path.append('Source Code')

try:
    from data_loader_redd import explore_available_meters, load_and_preprocess_redd

    print("Testing REDD data loader with actual redd.h5 file...")

    # Test exploring meters
    print("\nExploring available meters in redd.h5...")
    meters = explore_available_meters("redd.h5")

    if meters:
        print("Available meters in REDD dataset:")
        for building_num, meters_dict in meters.items():
            print(f"  Building {building_num}:")
            for meter_num, meter_name in meters_dict.items():
                print(f"    Meter {meter_num}: {meter_name}")

        # Test loading data for the first building and first meter
        first_building = list(meters.keys())[0]
        first_meter = list(meters[first_building].keys())[0]

        print(f"\nLoading data for Building {first_building}, Meter {first_meter}...")
        data_dict = load_and_preprocess_redd(
            "redd.h5",
            building_number=first_building,
            meter_number=first_meter,
            window_size=100,
            target_size=1
        )

        print("\n✅ Data loaded successfully!")
        print(f"Meter: {data_dict['meter_name']}")
        print(f"Training batches: {len(data_dict['train_loader'])}")
        print(f"Validation batches: {len(data_dict['val_loader'])}")
        print(f"Test batches: {len(data_dict['test_loader'])}")

    else:
        print("No meters found in the REDD dataset.")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()