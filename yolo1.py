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

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    # Process each track
    for track in tracks:
        # Check if the detected object is a person
        if 'label' in track.names and track.names['label'] == 'person':
            identifier = track.id

            # Take a snapshot of the entire frame
            snapshot_filename = os.path.join(snapshot_dir, "snapshot_identifier_{}.png".format(identifier))
            try:
                cv2.imwrite(snapshot_filename, im0)
                print("Snapshot saved:", snapshot_filename)
            except Exception as e:
                print("Error saving snapshot:", e)

    im0 = counter.start_counting(im0, tracks)
    cv2.imshow("Result", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
