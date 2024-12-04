import os
import json
import re


class Tracklet:
    def __init__(
        self, id, category, appearance=None, motion=None, trajectory=[]
    ):
        self.id = id
        self.appearance = appearance
        self.motion = motion
        self.trajectory = trajectory
        self.category = category

    def set_appearance(self, appearance):
        self.appearance = appearance

    def set_motion(self, motion):
        self.motion = motion

    def add_trajectory(self, time, coordinates):
        self.trajectory.append((time, coordinates))

    def clear_trajectory(self):
        self.trajectory = []

    def generate_description(self, trajectory_only=False):
        # assert len(self.trajectory) > 0
        if self.appearance is None:
            self.appearance = "the appearance is unknown"
        if self.motion is None:
            self.motion = "the motion is unknown"

        if not trajectory_only:
            des = f"{self.id}th instance, {self.category}, {self.appearance}, {self.motion}, the trajectory is (coordinate represented in the form of (x1, y1, x2, y2)): "
        else:
            des = "the trajectory is (coordinate represented in the form of (x1, y1, x2, y2)): "

        for t, c in self.trajectory[:2]:
            des += f"at {t} seconds, ({int(c[0])},{int(c[1])},{int(c[2])},{int(c[3])}), "

        des = des[:-2] + "..."

        return des


def generate_instance_bbox_json(folder_path, output_json_path):
    instance_dict = {}

    mask_pattern = re.compile(r'mask_(\d+)')

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                match = mask_pattern.search(filename)
                if match:
                    frame_number = match.group(1)
                    new_frame_name = f"{frame_number}.jpg"

                for instance_id, instance_info in data.get('labels', {}).items():
                    bbox = {
                        "frame_name": new_frame_name,
                        "bbox": [instance_info['x1'], instance_info['y1'], instance_info['x2'], instance_info['y2']]
                    }
                    
                    category = instance_info['class_name']
                    if instance_id not in instance_dict:
                        instance_dict[instance_id] = {
                            "category": category,
                            "bboxes": []
                        }
                    
                    instance_dict[instance_id]["bboxes"].append(bbox)

    sorted_instance_dict = {}
    for instance_id, data in instance_dict.items():
        sorted_bboxes = sorted(data["bboxes"], key=lambda x: int(re.findall(r'\d+', x['frame_name'])[0]))
        sorted_instance_dict[instance_id] = {
            "category": data["category"],
            "bboxes": sorted_bboxes
        }
    
    with open(output_json_path, 'w') as output_file:
        json.dump(sorted_instance_dict, output_file, indent=4)



def create_tracklets_from_json(json_file_path):
    all_tracklets = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)

        for instance_id, instance_info in data.items():
            category = instance_info['category']           
            tracklet = Tracklet(id=instance_id, category=category)

            for bbox_info in instance_info['bboxes']:
                frame_name = bbox_info['frame_name']
                bbox = bbox_info['bbox']
                frame_number = int(frame_name.split('.')[0])
                
                tracklet.add_trajectory(time=frame_number, coordinates=bbox)
            
            all_tracklets.append(tracklet)

    return all_tracklets
