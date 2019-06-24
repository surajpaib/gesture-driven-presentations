import xml.etree.ElementTree as ET

def normalize_point(point_to_normalize, avg_dist, avg_point):
    nx = point_to_normalize[0] - avg_point[0]
    nx = nx / avg_dist
    ny = point_to_normalize[1] - avg_point[1]
    ny = ny / avg_dist
    return [nx,ny]

def create_xml_for_keypoint(id, keypoint, avg_dist, avg_point, normalisation=False):
    keypoint_node = ET.Element('Keypoint')
    id_node = ET.SubElement(keypoint_node, 'ID')
    id_node.text = str(id)
    norm_point = [keypoint[0], keypoint[1]]
    if normalisation:
        norm_point = normalize_point([keypoint[0], keypoint[1]], avg_dist, avg_point)
    x_node = ET.SubElement(keypoint_node, 'X')
    x_node.text = str(norm_point[0])
    y_node = ET.SubElement(keypoint_node, 'Y')
    y_node.text = str(norm_point[1])
    conf_node = ET.SubElement(keypoint_node, 'Confidence')
    conf_node.text = str(keypoint[2])
    return keypoint_node