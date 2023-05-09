import xml.etree.ElementTree as ET


if __name__ == "__main__":
    for side in ["left", "right"]:
        urdf_filepath = f"../assets/urdf/shadow_robot/ur10e_shadow_hand_{side}.urdf"
        print("Loading", urdf_filepath)

        # Load the URDF file
        tree = ET.parse(urdf_filepath)
        root = tree.getroot()

        # Iterate over all joints
        for joint in root.iter("joint"):
            attrib = {}
            if joint.find("limit") is not None:
                for key, value in joint.find("limit").attrib.items():
                    attrib[key] = float(value)
            if joint.find("dynamics") is not None:
                for key, value in joint.find("dynamics").attrib.items():
                    attrib[key] = float(value)
            if len(attrib) != 0:
                print(f"\"{joint.attrib['name']}\": {attrib},")
