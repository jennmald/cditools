from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_excel_file(filename):
    """
    Parse Excel file and generate Ophyd device classes for each unique component.

    Parameters:
    -----------
    filename : str
        Path to the Excel file containing the component information.

    Returns:
    --------
    str
        Python code containing the generated Ophyd device classes.
    """
    # Read Excel file
    excel_df = pd.read_excel(filename, sheet_name="Motion axis summary", dtype=str)

    # Drop rows with NaN values in required columns
    required_columns = [
        "Component Name",
        "Component ID",
        "Full component name",
        "Full axis name",
        "Axis ID",
        "IOC Name",
    ]

    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in excel_df.columns]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    # Filter out rows with missing values in required columns
    excel_df = excel_df.dropna(subset=required_columns)

    # Group components by Component Name + Component ID
    components = defaultdict(list)

    for _, row in excel_df.iterrows():
        component_name = row["Component Name"]
        component_id = row["Component ID"]

        # Skip rows where Component Name is "MC"
        if component_name == "MC":
            logging.warning(
                "Skipping row with Component Name 'MC', Component ID: %s", component_id
            )
            continue

        full_component_name = row["Full component name"]
        full_axis_name = row["Full axis name"]
        axis_id = row["Axis ID"]

        # Get axis prefix by removing the full component name from the full axis name
        # This assumes the full_axis_name starts with full_component_name
        if full_axis_name.startswith(full_component_name):
            axis_prefix = full_axis_name[len(full_component_name) :]
        else:
            # If not, just use the full axis name (might need adjustment)
            axis_prefix = full_axis_name

        # Create unique key for components using Component Name + Component ID
        group_key = f"{component_name}_{component_id}"

        # Store component information
        components[group_key].append(
            {
                "component_name": component_name,
                "component_id": component_id,
                "full_component_name": full_component_name,
                "axis_prefix": axis_prefix,
                "axis_id": axis_id,
            }
        )

    # Generate Ophyd device classes
    return generate_ophyd_classes(components)


def generate_ophyd_classes(components):
    """
    Generate Ophyd device classes based on the parsed data.

    Parameters:
    -----------
    components : dict
        Dictionary containing component information.

    Returns:
    --------
    str
        Python code with Ophyd device class definitions.
    """
    code = [
        "#!/usr/bin/env python",
        "from ophyd import Device, Component as Cpt, EpicsMotor",
        "",
        "# Auto-generated Ophyd device classes",
        "",
    ]

    for _, axes in components.items():
        if not axes:
            continue

        # Use the first entry to get the component details
        first_axis = axes[0]
        component_name = first_axis["component_name"]
        component_id = first_axis["component_id"]
        full_component_name = first_axis["full_component_name"]

        # Start class definition - concatenate Component Name and Component ID for class name
        class_name = f"{component_name.replace(' ', '')}{component_id}"
        code.append(f"class {class_name}(Device):")
        code.append(f'    """Ophyd Device for {component_name} {component_id}"""')

        # Add component for each axis - without Component ID in name
        read_attrs = []
        for axis in axes:
            axis_id = axis["axis_id"]
            # Use only the Axis ID for the component name, not including Component ID
            motor_name = f"{axis_id}"
            motor_name_clean = motor_name.replace("-", "_").replace(" ", "_").lower()
            code.append(
                f"    {motor_name_clean} = Cpt(EpicsMotor, '-{axis['axis_prefix'].split('-')[-1]}')"
            )
            read_attrs.append(motor_name_clean)

        # Add default value for prefix
        code.append(f"    _default_prefix = '{full_component_name}'")
        code.append(f"    _default_read_attrs = {read_attrs}")
        code.append("")

        # Add instantiation code comment
        device_name = f"{component_name.lower()}_{component_id.lower()}"
        code.append(f"# Instantiate {device_name}")
        code.append(
            f"{device_name} = {class_name}(prefix='{full_component_name[:-1]}', name='{device_name}')"
        )
        code.append("")

    return "\n".join(code)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Excel file and generate Ophyd device classes."
    )
    parser.add_argument("filename", help="Path to the Excel file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output Python file (default: ophyd_devices.py)",
        default="ophyd_devices.py",
    )

    args = parser.parse_args()

    code = parse_excel_file(args.filename)

    with Path(args.output).open("w") as f:
        f.write(code)

    logging.info("Successfully generated Ophyd device classes in %s", args.output)

    return 0


if __name__ == "__main__":
    main()
