import xml.etree.ElementTree as ET

from frame_data import FrameData

class VideoData:
    def __init__(self):
        self._frames = []
        self._label = ""
        self._framerate = 0
    
    def get_keypoints_from_frame(self, frame_number):
        return (self._frames[frame_number]).keypoints

    def load_xml_file(self, xml_path):
        data_tree = ET.parse(xml_path)
        data_root = data_tree.getroot()
        label_node = data_root.find("Label")
        self._label = str(label_node.text)
        frame_rate_node = data_root.find("FPS")
        self._framerate = int(frame_rate_node.text)
        for frame_node in data_root.findall("Frame"):
            frame_data = FrameData()
            frame_data.load_from_xml(frame_node)
            self._frames.append(frame_data)

    @property
    def frames(self):
        return self._frames
    @property
    def label(self):
        return self._label
    @property
    def fps(self):
        return self._framerate

if __name__ == "__main__":
    v = VideoData()
    v.load_xml_file("C:\\Users\\AGANDO\\Videos\\videos\\Processing_Results_LNext_7\\2019-05-0211-25-02_mirror_x.mp4-reversed_7.xml")
    print(v.get_keypoints_from_frame(0)[0])
