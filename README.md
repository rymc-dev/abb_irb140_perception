# abb_irb140_perception

ROS 2 (Jazzy) perception for the ABB IRB140 with an eye-in-hand RGB-D camera
mounted on the gripper. A YOLO detector finds a ball in the RGB image, the
aligned depth image gives its range, and TF2 turns that into a position in
`base_link`. The result is published as a `base_link -> ball` transform, a
`vision_msgs/Detection3DArray`, and an RViz `MarkerArray`, for the downstream
motion/pick logic (e.g. `abb_irb140_motion_control`'s `ball_follow_node`, which
reads the `ball` TF).

The same node runs unchanged in Gazebo and on the real robot: it always
back-projects into `depth_camera_optical` and lets TF2 do the rest.

| | |
|---|---|
| Executable | `abb_irb140_perception_node` (`abb_irb140_perception/perception_node.py`) |
| ROS node | `/hybraut/ball_detector` (class `BallDetector`, namespace `hybraut`) |
| Launch file | `launch/abb_irb140_perception.launch.py` |

## Pipeline

```
RGB ──► YOLO ──► 2D ball box ─┐
                              ├─► pinhole back-projection ──► depth_camera_optical
depth (aligned to RGB) ───────┘                                      │
                                                                     │ TF2, looked up at the
                                                                     │ depth image's capture stamp
                                                                     ▼
                                          base_link ──► Detection3DArray
                                                    ──► MarkerArray
                                                    ──► TF  base_link -> ball
```

1. `ApproximateTimeSynchronizer` (`queue_size=5`, `slop=0.05` s) pairs an RGB
   and a depth image.
2. YOLO runs on the RGB frame; only classes in `target_classes` are kept.
3. For each box, the centre pixel is back-projected through the depth image
   (`hybraut_irb140.perception_geometry.backproject_pixel`). Depth is the
   **minimum valid** reading in a `(2·depth_window_px + 1)²` window; readings
   at or below `min_valid_depth_m` are ignored so the gripper's own fingers
   can't be mistaken for the ball.
4. The depth reading is the ball's near surface, so the point is pushed back
   along the optical Z axis by one estimated radius
   (`r = r_px · z / fx`, with `r_px` the mean half-size of the box) to
   approximate the sphere centre.
5. The point is stamped `camera_optical_frame` at the depth image's stamp and
   transformed to `base_frame` with TF2 (see [TF and timing](#tf-and-timing)).
6. Outputs are published (below). Ball orientation is always identity.

The frame the point is assigned to is always the `camera_optical_frame`
parameter, **not** the `frame_id` in `CameraInfo`. This matters in simulation,
where the camera's `CameraInfo` reports `depth_camera` (the body-convention
frame) rather than `depth_camera_optical`.

## Requirements

- **`hybraut_irb140`** (sibling package in this workspace). This node imports
  `hybraut_irb140.perception_geometry` (`backproject_pixel`, `depth_to_meters`)
  and, when `model_path` is empty, looks for
  `share/hybraut_irb140/weights/ball_yolo.pt`. `package.xml` in this package
  does not yet declare these dependencies, so build `hybraut_irb140` first.
- **Python:** `ultralytics` and `torch` (pip). A CUDA build of torch is needed
  for the launch file's default `device: "0"`.
- **ROS packages:** `cv_bridge`, `image_geometry`, `message_filters`,
  `tf2_ros`, `tf2_geometry_msgs`, `vision_msgs`, `visualization_msgs`.
- **A camera** publishing RGB, depth and `CameraInfo`. The depth image must be
  registered to the colour image, because pixel coordinates from the RGB box
  index straight into depth. Depth may be `32FC1` metres (Gazebo) or `16UC1`
  millimetres (RealSense-style); both are handled.
- **A TF tree** that connects `depth_camera_optical` to `base_link`, published
  from live `/joint_states`. In this rig that is
  `tool0 → pneumatic_gripper_base_link → depth_camera → depth_camera_optical`.
  The `depth_camera → depth_camera_optical` fixed rotation must be in the URDF.

### Model weights

With `model_path` empty the node uses `ball_yolo.pt` from the
`hybraut_irb140` share directory if it exists, otherwise `yolov8n.pt` (looked
up by ultralytics, so it is downloaded or read from the current directory).
A generic COCO model may or may not pick up the synthetic Gazebo sphere as
`sports ball`, so a fine-tuned model is more reliable. Set `model_path` to use
your own weights.

## Build and run

```bash
cd ~/ros2_ws
colcon build --packages-select abb_irb140_perception
source install/setup.bash

# Simulation (Gazebo clock)
ros2 launch abb_irb140_perception abb_irb140_perception.launch.py use_sim_time:=true

# Real robot (wall clock; this is the launch default)
ros2 launch abb_irb140_perception abb_irb140_perception.launch.py
```

`use_sim_time` defaults to `false`, so in simulation pass `use_sim_time:=true`
to put the node on the Gazebo clock like the rest of the stack.

The launch file sets `device: "0"` (first CUDA GPU). To run without a GPU, run
the executable directly and override it, but see the
[inference backlog note](#tf-and-timing) first:

```bash
ros2 run abb_irb140_perception abb_irb140_perception_node \
  --ros-args -p device:=cpu -p imgsz:=320
```

## Interface

### Subscribed

| Topic (default) | Type | Parameter |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | `rgb_topic` |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | `depth_topic` |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | `camera_info_topic` |
| `/tf`, `/tf_static` | | |

Nothing is published until the first `CameraInfo` arrives.

### Published

| Output | Type | Notes |
|---|---|---|
| `/hybraut/hybraut_irb140/ball_detections_3d` | `vision_msgs/Detection3DArray` | `frame_id = base_frame`. Box centre = ball centre; box size = estimated diameter; one hypothesis with `class_id` and YOLO score. |
| `/hybraut/hybraut_irb140/ball_markers` | `visualization_msgs/MarkerArray` | Green translucent spheres, namespace `marker_ns`. Markers for balls that disappeared are deleted. |
| `/hybraut/hybraut_irb140/ball_detections/image` | `sensor_msgs/Image` | RGB frame with boxes and labels; carries the RGB image's own header. |
| TF `base_link -> ball` | | Named `<tf_frame>` for one ball, `<tf_frame>_0`, `<tf_frame>_1`, … for several. Identity rotation. |

The output topic names are absolute and hard-coded; they are not affected by
the node namespace or remapped by parameters. Publishers can be turned off
individually with `publish_detections_3d`, `publish_markers` and
`publish_annotated`.

Detection3D, ADD markers and the ball TF are all stamped with the **depth**
image's timestamp (the instant the position was measured).

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `rgb_topic` | `/camera/color/image_raw` | RGB image topic. |
| `depth_topic` | `/camera/depth/image_rect_raw` | Depth image topic (must be registered to RGB). |
| `camera_info_topic` | `/camera/color/camera_info` | Intrinsics for the RGB/depth pixel grid. |
| `base_frame` | `base_link` | Target frame for all outputs. |
| `camera_optical_frame` | `depth_camera_optical` | Frame the back-projected point is assigned to. Must be the optical frame (x right, y down, z forward). |
| `model_path` | `""` | YOLO weights. Empty = packaged `ball_yolo.pt`, else `yolov8n.pt`. |
| `device` | `cpu` (launch file: `"0"`) | Ultralytics device: `cpu`, or a CUDA index as a string. |
| `confidence_threshold` | `0.15` | YOLO confidence. |
| `iou_threshold` | `0.45` | YOLO NMS IoU. |
| `imgsz` | `640` | YOLO inference size. |
| `max_detections` | `10` | Max boxes per frame. |
| `target_classes` | `["sports ball", "ball"]` | Class names to keep. Empty = all classes. |
| `depth_window_px` | `2` | Half-width of the depth sampling window (min-reduced). |
| `min_valid_depth_m` | `0.05` | Depth at or below this is ignored (gripper self-occlusion guard). |
| `min_score_for_3d` | `0.0` | Boxes scoring below this are drawn but not localised. |
| `tf_frame` | `ball` | Child frame name for the ball TF. |
| `marker_ns` | `ball` | Marker namespace. |
| `marker_lifetime` | `0.5` | Marker lifetime in seconds. |
| `marker_alpha` | `0.5` | Marker opacity. |
| `publish_annotated` | `true` | Publish the annotated RGB image. |
| `publish_markers` | `true` | Publish the RViz markers. |
| `publish_detections_3d` | `true` | Publish the `Detection3DArray`. |
| `tf_timeout` | `0.15` | Seconds TF2 may wait for the transform at the exact capture stamp. |
| `max_tf_staleness` | `0.03` | Max age (s) of the "latest TF" fallback relative to the capture stamp before the detection is dropped. `<= 0` disables the fallback. |

## TF and timing

The camera moves with the arm, so a point measured at time *t* must be
transformed with the camera pose at time *t*. Any other pose gives an error
proportional to how far the arm moved in between. `_transform_point()` handles
this in two steps:

1. Look up the transform at the depth image's exact stamp, waiting up to
   `tf_timeout`.
2. If TF hasn't caught up (`ExtrapolationException`), fall back to the latest
   available transform, but **only** if that transform is within
   `max_tf_staleness` of the capture stamp. Otherwise the detection is dropped
   and a `Dropping ball detection: ... ms from the image's capture stamp`
   warning is logged.

A dropped frame publishes no TF, Detection3D or ADD marker. A missing
detection is far safer for pick logic than a confidently wrong one, so
**consumers should check the age of the `ball` TF** rather than assume it is
fresh (`abb_irb140_motion_control`'s `ball_follow_node` and `move_to_ball_node`
do this with `max_tf_age_sec`).

`max_tf_staleness` is sized as
`acceptable_position_error / max_tool_speed`. At an eye-in-hand approach speed
of 0.3–0.5 m/s, 30 ms bounds the error to about 1–1.5 cm. Retune it to your
rig's real end-effector speed.

### Why the node is built this way

These three problems were found in turn while chasing a ball position that
was right when the arm was still and wrong (by centimetres) while it moved.
Each one explains a piece of the current code.

1. **Unbounded latest-TF fallback.** The original fallback accepted any
   transform, however old. It looked fine at rest (every transform was
   identical) and failed only during motion. Diagnosed from a bag of `/tf` and
   `/joint_states` with a stationary ball: the published position swung about
   7.7 cm (x), 22.3 cm (y) and 7.3 cm (z), tracking exactly the periods when
   the joints were moving. Fixed with `max_tf_staleness`, and by stamping every
   output from the depth image rather than mixing in the RGB stamp (the two
   can differ by up to `slop`, 50 ms).
2. **CPU inference backlog.** With `device: cpu`, YOLO competed with Gazebo
   for CPU and the node fell steadily behind (a constant ~4.7 s of staleness
   on every frame). `ApproximateTimeSynchronizer` queues frames rather than
   dropping them, so this looked like TF lag. The launch file now runs
   inference on the GPU. If you must use CPU, lower `imgsz`.
3. **Executor starving the TF listener.** With plain `rclpy.spin()`, the image
   callback and `tf2_ros.TransformListener` shared one thread. Under the
   heavier load of a real MoveIt trajectory (as opposed to a manual joint
   jog) the TF buffer could lag by ~4.7 s even though `/joint_states` was
   arriving at 64–102 Hz. `main()` now uses
   `MultiThreadedExecutor(num_threads=4)` so the listener's own callback
   group gets a separate lane. The image callback is still serialised by
   its default mutually-exclusive callback group; the only shared state is the
   thread-safe `tf2_ros.Buffer`.

To check a change end to end, run a real MoveIt-executed pick (not just a
manual jog) against a stationary ball and confirm both that
`Dropping ball detection` warnings appear only during genuinely fast motion,
and that the ball is not disturbed. To reset the simulated ball:

```bash
gz service -s /world/robot_lab/set_pose --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean \
  --req 'name: "my_sphere", position: {x: 0.55, y: 0.165182, z: 1.086870}, orientation: {w: 1.0}'
gz model -m my_sphere -p   # read back the ground-truth pose
```

## Validation

`notebooks/validate_transform_pipeline.ipynb` checks the static
`depth_camera_optical -> base_link` chain and the detector geometry against
Gazebo ground truth (`gz model -m my_sphere -p`). It is a **stationary-arm**
test and does not cover motion, which is what the TF handling above is for.

- Test 1 projects the ground-truth ball into the image through forward
  kinematics and compares the depth there (no detector involved).
- Test 2 runs YOLO plus the same back-projection and radius correction as the
  node and reports the world-frame error against a 5 cm tolerance.
- `notebooks/perception_inputs_multi/pos1..pos4/` are tracked single-frame
  captures (`rgb.png`, `depth.npz`, `camera_info.yaml`, ground truth) at four
  other ball positions. `notebooks/perception_inputs/` (a ~3 GB bag) is
  gitignored, so only the multi-position section runs from a fresh clone.

The notebook predates the current file layout: it refers to
`abb_irb140_perception_node.py` (now `perception_node.py`) and runs YOLO on
CPU.

## Local-only artefacts (gitignored)

`docs/` and `bags/` are not in the repository. On the development machine:

- `docs/initial_perception_to_transform_pipeline.mp4`: the pipeline working
  with the arm stationary.
- `docs/perception_tranformation_error_while_in_motion.mp4`: the motion bug.
- `docs/perception_fix.mp4`: after the fix.
- `docs/real_robot_perception_and_translation_demo.mp4`: the real robot.
- `bags/rosbag2_2026_09_17-17_14_02/` and `bags/ball_ground_truth.txt`: the
  `/tf`, `/tf_static`, `/joint_states` bag and ground truth used to diagnose
  the motion bug. It has no camera data, so it cannot drive the node.

## Package layout

```
abb_irb140_perception/perception_node.py   # BallDetector node and main()
launch/abb_irb140_perception.launch.py     # sets device=0 and use_sim_time
notebooks/                                 # transform-pipeline validation
test/                                      # ament copyright / flake8 / pep257
```

## Known issues and follow-ups

- `package.xml` still has placeholder description and license and does not
  list runtime dependencies (`hybraut_irb140`, `rclpy`, the message packages,
  `ultralytics`/`torch`).
- `abb_irb140_bringup` starts `hybraut_irb140`'s own
  `hybraut_irb140_ball_detector`, not this node. That detector still has the
  unbounded latest-TF fallback and spins on plain `rclpy.spin()`, so it is
  exposed to problems 1 and 3 above. It also needs a
  `camera_optical_frame_override` for the reason given in
  [Pipeline](#pipeline). Do not run both detectors at once: they share the node
  name `/hybraut/ball_detector`, the output topics and the `ball` TF.
- Detections are per-frame with no tracking or filtering; the ball position
  is not smoothed across frames.
