import xml.etree.ElementTree as ET

class FrameData:
    def __init__(self):
        self._keypoints = []
        self._avg_point = [0,0]
        self._avg_dist = 0
    def load_from_xml(self, frame_node : ET.Element):
        avg_x_node = frame_node.find("Avg_x")
        self._avg_point[0] = int(avg_x_node.text)
        avg_y_node = frame_node.find("Avg_y")
        self._avg_point[1] = int(avg_y_node.text)
        avg_dist_node = frame_node.find("Avg_dist")
        self._avg_dist = float(avg_dist_node.text)
        for keypoint_node in frame_node.findall("Keypoint"):
            keypoint = [0,0,0]
            x_node = keypoint_node.find("X")
            keypoint[0] = float(x_node.text)
            y_node = keypoint_node.find("Y")
            keypoint[1] = float(y_node.text)
            confidence_node = keypoint_node.find("Confidence")
            keypoint[2] = float(confidence_node.text)
            self._keypoints.append(keypoint)
    
    @property
    def keypoints(self):
        return self._keypoints
    @property
    def avg_point(self):
        return self._avg_point
    @property
    def avg_dist(self):
        return self._avg_dist