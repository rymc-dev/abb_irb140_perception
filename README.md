# abb_irb140_perception

ROS 2 (Jazzy) perception node for detecting a ball with an eye-in-hand RGB-D
camera mounted on the ABB IRB140's gripper, and publishing its 3D position as
a `base_link -> ball` TF transform (plus `visualization_msgs/MarkerArray` and
`vision_msgs/Detection3DArray`) for downstream motion/pick logic.

Node: `abb_irb140_perception_node` (`abb_irb140_perception/perception_node.py`),
launched via `launch/abb_irb140_perception.launch.py`.

## Known issue: ball TF jumped to the wrong position while the arm was moving

**Symptom:** the published `base_link -> ball` transform was correct once the
arm stopped and settled, but became wrong — sometimes by several centimetres —
while the arm was actively moving, which broke downstream motion/pick logic
that consumed it.

A video demonstrating the problem (ball marker visibly detaching from the
true ball position during arm motion, then snapping back once the arm stops)
is at
[`docs/perception_tranformation_error_while_in_motion.mp4`](docs/perception_tranformation_error_while_in_motion.mp4).
An earlier video showing the intended perception -> transform pipeline
working (arm stationary) is at
[`docs/initial_perception_to_transform_pipeline.mp4`](docs/initial_perception_to_transform_pipeline.mp4).

### Diagnosis

To debug this without needing to reproduce it live, a rosbag was recorded
covering `/tf`, `/tf_static`, and `/joint_states` while the ball detector ran
against a physically stationary ball
(`bags/rosbag2_2026_09_17-17_14_02/`), together with the ball's ground-truth
pose in the sim (`bags/ball_ground_truth.txt`).

Since the ball never actually moved, its true `base_link`-frame position
should have been constant throughout the recording. Instead, the published
`base_link -> ball` transform swung by **~7.7 cm in x, ~22.3 cm in y, and
~7.3 cm in z** over the ~20 s recording — and each swing tracked periods
where `/joint_states` was actively changing (the arm moving), snapping back
to a stable value every time the arm paused. That is the "perfect when
stopped, wrong while moving" symptom exactly, and it's a systematic
computation error correlated with motion, not sensor noise.

### Root cause

The camera is eye-in-hand
(`tool0 -> pneumatic_gripper_base_link -> depth_camera -> depth_camera_optical`,
all published dynamically by `robot_state_publisher` from `/joint_states`), so
the camera's pose relative to `base_link` changes continuously while the arm
moves. Each detected ball position has to be transformed from the camera's
optical frame into `base_link` using the TF tree *as it was at the instant
the depth image was captured* — using any other instant's pose introduces a
position error proportional to how far the arm moved in between.

`_transform_point()` in `perception_node.py` looked up that transform at the
depth image's exact capture timestamp, but if TF publication hadn't caught up
yet (`tf2_ros.ExtrapolationException` — common under simulation/CPU load), it
silently fell back to whatever the **latest available** transform was, with
no check on how stale that fallback actually was. While the arm was
stationary this fallback was harmless (every available transform was
identical, since the pose wasn't changing) — which is exactly why the bug was
invisible at rest and only appeared during motion.

A smaller, compounding issue: the point used for the TF lookup was stamped
from the depth image (`depth_msg.header.stamp`), but the *published*
TF/marker/Detection3D outputs were stamped from the RGB image
(`rgb_msg.header.stamp`) instead. RGB and depth are only approximately
time-synced (`message_filters.ApproximateTimeSynchronizer(..., slop=0.05)`),
so these could differ by up to 50 ms — a second, smaller source of the same
class of temporal-mismatch error.

### Fix

Both changes are in `abb_irb140_perception/perception_node.py`:

1. **Bounded the "latest transform" fallback.** `_transform_point()` now
   only accepts the fallback transform if it is within a new
   `max_tf_staleness` parameter (default `0.03` s / 30 ms) of the point's
   true capture time — measured against the *actual* evaluation time TF2
   stamps the transformed point with, not just how long the lookup took. If
   the fallback is staler than that, the detection is dropped (no
   TF/marker/Detection3D published for that frame) instead of published with
   a wrong, motion-proportional position. A momentarily missing detection is
   far safer for downstream motion/pick logic than a confidently wrong one.
   Also raised `tf_timeout` from `0.1` s to `0.15` s, giving TF2's own
   bounded wait (`Buffer.transform(..., timeout=...)` genuinely blocks until
   the requested stamp becomes available, not just an instant check) more
   room to absorb ordinary publish jitter before falling back at all.

   `max_tf_staleness` is sized as roughly
   `acceptable_position_error / max_expected_tool_speed`: at a plausible
   eye-in-hand approach speed of ~0.3-0.5 m/s, 30 ms of staleness bounds the
   induced position error to ~1-1.5 cm. Retune it against this rig's actual
   max end-effector speed during approach/pick motions if needed. Setting it
   to `0` disables the fallback entirely (strict "exact stamp within
   `tf_timeout`, or drop the detection").

2. **Made the timestamp used for the ball's position consistent.** The point
   backprojected from the depth image, the TF lookup, and everything
   published for that detection (`Detection3DArray.header.stamp`,
   `Marker.header.stamp`, `TransformStamped.header.stamp`) now all use
   `depth_msg.header.stamp` instead of mixing in `rgb_msg.header.stamp` —
   removing the up-to-50ms RGB/depth sync gap as a second source of the same
   error class. (The annotated RGB debug image is unaffected and still
   correctly carries the RGB frame's own header.)

New parameters (both on `abb_irb140_perception_node`):

| Parameter | Default | Meaning |
|---|---|---|
| `tf_timeout` | `0.15` (was `0.1`) | Seconds to let TF2 wait for the transform to become available at the exact capture stamp before falling back. |
| `max_tf_staleness` | `0.03` | Seconds the fallback ("latest available") transform's own evaluation time may drift from the detection's true capture time before the detection is dropped instead of published. `<= 0` disables the fallback entirely. |

### A second, separate problem the fix surfaced: CPU-bound inference backlog

After deploying the fix above, live testing showed *every* detection being
dropped with a steady ~4.7 s staleness — far larger than the sub-second
publish jitter seen in the original diagnostic bag, and not growing/shrinking
over time. A constant, steady-state gap like that is the signature of the
node's own processing falling behind the incoming frame rate, not of TF
publishing being briefly behind: the `device` parameter defaults to `"cpu"`
(`perception_node.py`), so YOLO inference at `imgsz=640` was running on CPU
on the same machine as the Gazebo simulation, competing for cycles the sim
also needed. The `ApproximateTimeSynchronizer(queue_size=5)` lets frames
queue rather than drop, so each processed frame was already several seconds
stale by the time `_transform_point()` ran, regardless of how fast the arm
was actually moving at that instant.

Confirmed a GPU was available and unused (`torch.cuda.is_available() ==
True`, an NVIDIA RTX 4050 Laptop GPU), so `launch/abb_irb140_perception.launch.py`
now passes `parameters=[{"device": "0"}]` to route inference to the GPU
instead of the CPU. **Rebuild before this takes effect**
(`colcon build --packages-select abb_irb140_perception`, then re-source), and
watch the "Dropping ball detection" staleness figures in the log — they
should now be tens of milliseconds during genuinely fast motion, not
seconds. If a GPU isn't available on a given machine, `device` can be
overridden back to `"cpu"` via `--ros-args -p device:=cpu`, but expect the
same backlog symptom to return, or reduce `imgsz` to cut per-frame inference
cost instead.

This is an important distinction for writing this up: the TF-staleness fix
(`max_tf_staleness`) didn't cause this backlog — it was always there,
silently absorbed by the old unbounded fallback, which is exactly what made
the original bug ("perfect when stopped, wrong while moving") so easy to
miss. The fix didn't introduce a new failure mode; it turned an invisible
one into a visible, diagnosable one.

### A third, deeper problem: the node's own executor could starve its TF listener

Even after switching inference to the GPU, live testing during *real*
MoveIt-executed pick trajectories (as opposed to simple manual joint jogs)
still reproduced a large, consistent **~4.7 s** staleness — reproducing even
from a cold restart of the whole stack, so it wasn't gradual drift or GPU
warm-up. With `max_tf_staleness` in place this correctly dropped every
detection during the stale window rather than publishing a wrong position,
but with no trustworthy detection available during the pick's critical
approach window, the grasp executed against stale/cached information and
knocked the ball off the table.

Direct measurement against the live system narrowed this down precisely:
- `ros2 topic hz /joint_states`, sampled during one of these exact stale
  windows, stayed a healthy 64-102 Hz throughout — the underlying joint
  state / TF *source* was not gapping for anywhere near 4.7 s.
- The node's own detection-output cadence stayed ~170-200 ms between
  messages throughout the same window — no single call to `_process()` was
  itself blocking for ~4.7 real seconds; many fast calls in a row were all
  seeing the same stale "latest" TF.
- A standalone probe node (its own process, own `tf2_ros.Buffer`/
  `TransformListener`, subscribed to the same `/tf` and depth image topics)
  measured only 10-50 ms of lag during two manual jog motions. Only a real,
  fully-loaded MoveIt trajectory execution reproduced the multi-second stall
  inside the actual perception node.

The root cause: `main()` spun the node with plain `rclpy.spin(node)`, i.e.
rclpy's default `SingleThreadedExecutor`. `tf2_ros.TransformListener`
(constructed at `perception_node.py` ~line 504-511) deliberately places its
`/tf`/`/tf_static` subscriptions in their own `ReentrantCallbackGroup` — its
own source comments explain this exists specifically so TF updates aren't
blocked by other callbacks — but that separation only pays off under a
multi-threaded executor. Under a single thread, the image-processing
callback (in the node's implicit default `MutuallyExclusiveCallbackGroup`)
and the TF listener's callback still serialize on the same OS thread
regardless of callback group. A real MoveIt trajectory generates far more
ROS 2 executor activity (controller feedback, planning-scene/octomap
traffic, etc.) than a manual jog, consistent with the single thread
occasionally starving the TF listener's callback group for several seconds
under that heavier load while still getting around to servicing image
callbacks — matching every measurement above.

**Fix:** `main()` now spins the node with an explicit
`rclpy.executors.MultiThreadedExecutor(num_threads=4)` instead of plain
`rclpy.spin(node)`. No other code changes were needed — the TF listener
already had its own callback group, and every other subscription/publisher
in this node already shared the default group with each other, so this was
exactly the two-lane split needed. `num_threads=4` matches the convention
already used by other hand-rolled nodes in the sibling `hybraut_irb140`
package (which pair `ReentrantCallbackGroup` with an explicit
`MultiThreadedExecutor(num_threads=4)`), and comfortably covers this node's
two real concurrency lanes without oversubscribing a process that also does
GPU-bound inference. The only shared state between the two lanes is
`self._tf_buffer` itself, which `tf2_ros`'s core is specifically designed to
be thread-safe for (concurrent reads against background writes) — the exact
pattern `TransformListener`'s own `spin_thread=True` option relies on.

### Verifying the fix

The bug reproduction bag only contains `/tf`/`/tf_static`/`/joint_states` (no
raw camera data), so it can't drive the node end-to-end, but it's enough to
sanity-check the timing assumption: replaying it and checking that the
staleness measured by the new fallback logic spikes exactly during the same
arm-moving windows that produced the original 7.7/22.3/7.3 cm swings.

Full end-to-end verification needs a live/sim run: launch
`abb_irb140_perception.launch.py`, move the arm through a similar sweep with
a stationary ball, record a new `/tf` + `/joint_states` bag the same way, and
confirm the `base_link -> ball` position no longer swings by cm-scale amounts
during motion. Watch the node's log for the new "Dropping ball detection"
warnings — they should appear only during genuinely fast motion, not
constantly; if they're constant, `tf_timeout` and/or `max_tf_staleness` need
to be raised.

The real test for the executor fix is a full MoveIt-executed pick attempt
(not just a manual jog), since that's what reproduced the ~4.7 s stall.
Reset the ball to its spawn pose if needed
(`gz service -s /world/robot_lab/set_pose --reqtype gz.msgs.Pose --reptype
gz.msgs.Boolean --req 'name: "my_sphere", position: {x: 0.55, y: 0.165182,
z: 1.086870}, orientation: {w: 1.0}'`) and confirm via `gz model -m my_sphere
-p` that a pick attempt no longer knocks it off the table.

### Known follow-up (not yet applied)

A separate, structurally similar node in the sibling `hybraut_irb140`
package (`hybraut_irb140_ball_detector`, used by
`abb_irb140_bringup`'s launch file) has the same unguarded
latest-transform-fallback pattern in its own `_transform_point()`, and would
benefit from the same bounded-staleness fix if/when that node is in active
use. It does not have the depth/RGB stamp-mismatch issue described above —
it already uses one consistent timestamp source. Worth checking separately
whether it spins on a `SingleThreadedExecutor` too, since it would be
susceptible to the same TF-listener-starvation issue described above if so.
