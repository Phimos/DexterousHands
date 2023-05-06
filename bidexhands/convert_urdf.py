import xml.etree.ElementTree as ET
import math


def flip(link: ET.Element, visual: bool = True, collision: bool = False) -> None:
    """Flip visual or collision of attached meshes

    Switch meshes from Z-up left-handed system to Y-up right-handed system

    Args:
        link (ET.Element): Link to be flipped
        visual (bool, optional): Switch visual mesh. Defaults to True.
        collision (bool, optional): Switch collision mesh. Defaults to False.
    """
    if visual and link.find("visual") is not None and link.find("visual").find("origin") is not None:
        origin = link.find("visual").find("origin")
        roll, pitch, yaw = origin.attrib["rpy"].split(" ")
        roll = str(float(roll) + math.pi / 2)
        origin.attrib["rpy"] = " ".join([roll, pitch, yaw])
    if collision and link.find("collision") is not None and link.find("collision").find("origin") is not None:
        origin = link.find("collision").find("origin")
        roll, pitch, yaw = origin.attrib["rpy"].split(" ")
        roll = str(float(roll) + math.pi / 2)
        origin.attrib["rpy"] = " ".join([roll, pitch, yaw])


if __name__ == "__main__":
    for side in ["left", "right"]:
        urdf_filepath = f"../assets/urdf/shadow_robot/ur10e_shadow_hand_{side}.urdf"

        # Load the URDF file
        tree = ET.parse(urdf_filepath)
        root = tree.getroot()

        # Iterate over all links
        for link in root.iter("link"):
            if link.attrib["name"].startswith("lh_") or link.attrib["name"].startswith("rh_"):
                if link.find("visual") is None or link.find("visual").find("origin") is None:
                    continue
                if "distal" in link.attrib["name"] or "forearm" in link.attrib["name"]:
                    flip(link, visual=True, collision=True)
                if "palm" in link.attrib["name"]:
                    flip(link, visual=True, collision=False)
            else:
                flip(link, visual=True, collision=False)

        # Save the modified URDF file
        tree.write(urdf_filepath)
