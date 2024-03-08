import cv2
import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("vid1.mp4")
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(584, 809), (1217, 809), (1217, 155), (584, 155)]

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

# Directory to save snapshots
snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

# Dictionary to store processed identifiers or persons
processed_identifiers = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    # Process each track
    for track in tracks:
        # Check if the detected object is a person
        if 'label' in track.names and track.names['label'] == 'person' and track.id not in processed_identifiers:
            identifier = track.id

            # Take a snapshot in the region of interest
            snapshot = im0[region_points[2][1]:region_points[0][1], region_points[0][0]:region_points[1][0]].copy()

            # Save the snapshot in the "snapshots" directory
            snapshot_filename = os.path.join(snapshot_dir, "snapshot_identifier_{}.png".format(identifier))
            try:
                cv2.imwrite(snapshot_filename, snapshot)
                print("Snapshot saved:", snapshot_filename)
            except Exception as e:
                print("Error saving snapshot:", e)

            # Mark this identifier as processed
            processed_identifiers.add(identifier)

    im0 = counter.start_counting(im0, tracks)

cap.release()
cv2.destroyAllWindows()
