#!/usr/bin/env python3
"""
ball_detector.py

Perception node for the ABB IRB140.

Pipeline:

    RGB image
        |
        v
      YOLO
        |
        v
    2D ball detection
        |
        v
    Aligned depth
        |
        v
    Pinhole back-projection
        |
        v
depth_camera_optical
        |
        | TF2
        v
    base_link
        |
        +--> Detection3DArray
        +--> RViz MarkerArray
        +--> ball TF
        +--> annotated image


Coordinate frames
-----------------

The Gazebo camera has two conceptual frames:

    depth_camera
        Body-style REP-103 frame:
            x = forward
            y = left
            z = up

    depth_camera_optical
        REP-103 optical frame:
            x = right
            y = down
            z = forward

The physical RealSense should expose the same optical frame:

    depth_camera_optical

The pinhole camera model produces coordinates in the OPTICAL frame.

Therefore this node always assigns the result of the depth
back-projection to:

    depth_camera_optical

TF2 is then responsible for transforming:

    depth_camera_optical -> base_link

This means the perception algorithm is identical in simulation
and on the physical robot.

IMPORTANT
---------

The TF tree must contain:

    depth_camera
        |
        | fixed rotation
        v
    depth_camera_optical
        |
        | camera mounting transform
        v
    ... robot links ...
        |
        v
    base_link

For simulation, the Gazebo URDF should therefore publish the
depth_camera -> depth_camera_optical fixed transform.

For the physical RealSense, the corresponding RealSense TF
should provide depth_camera_optical relative to the robot/tool.

Outputs
-------

/hybraut/hybraut_irb140/ball_detections_3d
    vision_msgs/Detection3DArray

/hybraut/hybraut_irb140/ball_markers
    visualization_msgs/MarkerArray

/hybraut/hybraut_irb140/ball_detections/image
    Annotated RGB image

TF:

    base_link -> ball

or, for multiple detections:

    base_link -> ball_0
    base_link -> ball_1
    ...

Requires:

    ultralytics
    torch
"""

import os
from typing import List, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.time import Time

from rcl_interfaces.msg import ParameterDescriptor

from message_filters import (
    ApproximateTimeSynchronizer,
    Subscriber,
)

from ament_index_python.packages import (
    get_package_share_directory,
)

import cv2

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import tf2_ros
from tf2_ros import (
    Buffer,
    TransformBroadcaster,
    TransformListener,
)

import tf2_geometry_msgs  # noqa: F401

from sensor_msgs.msg import (
    Image,
    CameraInfo,
)

from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
)

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)

from hybraut_irb140.perception_geometry import (
    backproject_pixel,
    depth_to_meters,
)


class BallDetector(Node):

    def __init__(self) -> None:

        super().__init__(
            "ball_detector",
            namespace="hybraut",
        )

        # ==============================================================
        # Camera topics
        # ==============================================================

        self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        )

        self.declare_parameter(
            "depth_topic",
            "/camera/depth/image_rect_raw",
        )

        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )

        # ==============================================================
        # TF frames
        # ==============================================================

        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        # IMPORTANT:
        #
        # The result of PinholeCameraModel back-projection is an
        # optical-frame coordinate:
        #
        #     x = right
        #     y = down
        #     z = forward
        #
        # Therefore this MUST be the optical frame, not the Gazebo
        # body-style depth_camera frame.
        #
        # Both simulation and real hardware should ultimately provide:
        #
        #     depth_camera_optical
        #
        self.declare_parameter(
            "camera_optical_frame",
            "depth_camera_optical",
        )

        # ==============================================================
        # YOLO parameters
        # ==============================================================

        self.declare_parameter(
            "model_path",
            "",
        )

        self.declare_parameter(
            "device",
            "cpu",
            ParameterDescriptor(
                dynamic_typing=True
            ),
        )

        self.declare_parameter(
            "confidence_threshold",
            0.15,
        )

        self.declare_parameter(
            "iou_threshold",
            0.45,
        )

        self.declare_parameter(
            "imgsz",
            640,
        )

        self.declare_parameter(
            "max_detections",
            10,
        )

        self.declare_parameter(
            "target_classes",
            ["sports ball", "ball"],
        )

        # ==============================================================
        # Depth parameters
        # ==============================================================

        self.declare_parameter(
            "depth_window_px",
            2,
        )

        self.declare_parameter(
            "min_valid_depth_m",
            0.05,
        )

        self.declare_parameter(
            "min_score_for_3d",
            0.0,
        )

        # ==============================================================
        # Output parameters
        # ==============================================================

        self.declare_parameter(
            "tf_frame",
            "ball",
        )

        self.declare_parameter(
            "marker_ns",
            "ball",
        )

        self.declare_parameter(
            "marker_lifetime",
            0.5,
        )

        self.declare_parameter(
            "marker_alpha",
            0.5,
        )

        self.declare_parameter(
            "publish_annotated",
            True,
        )

        self.declare_parameter(
            "publish_markers",
            True,
        )

        self.declare_parameter(
            "publish_detections_3d",
            True,
        )

        self.declare_parameter(
            "tf_timeout",
            0.15,
        )

        # How far the "latest available transform" fallback's own
        # timestamp may drift from the detection's true capture time
        # before the detection is dropped instead of published. The
        # camera is eye-in-hand, so a stale fallback transform composed
        # with a camera-frame point produces a position error
        # proportional to how far the arm moved in that gap -- bounding
        # it keeps a momentary dropped detection instead of a
        # confidently wrong one. <= 0 disables the fallback entirely.
        self.declare_parameter(
            "max_tf_staleness",
            0.03,
        )

        # ==============================================================
        # Read parameters
        # ==============================================================

        self._base_frame = self.get_parameter(
            "base_frame"
        ).value

        self._camera_optical_frame = (
            self.get_parameter(
                "camera_optical_frame"
            ).value
        )

        self._device = str(
            self.get_parameter(
                "device"
            ).value
        )

        self._conf = self.get_parameter(
            "confidence_threshold"
        ).value

        self._iou = self.get_parameter(
            "iou_threshold"
        ).value

        self._imgsz = self.get_parameter(
            "imgsz"
        ).value

        self._max_det = self.get_parameter(
            "max_detections"
        ).value

        self._depth_window = self.get_parameter(
            "depth_window_px"
        ).value

        self._min_valid_depth_m = (
            self.get_parameter(
                "min_valid_depth_m"
            ).value
        )

        self._min_score_for_3d = (
            self.get_parameter(
                "min_score_for_3d"
            ).value
        )

        self._tf_frame = self.get_parameter(
            "tf_frame"
        ).value

        self._marker_ns = self.get_parameter(
            "marker_ns"
        ).value

        self._marker_lifetime = (
            self.get_parameter(
                "marker_lifetime"
            ).value
        )

        self._marker_alpha = (
            self.get_parameter(
                "marker_alpha"
            ).value
        )

        self._publish_annotated = (
            self.get_parameter(
                "publish_annotated"
            ).value
        )

        self._publish_markers = (
            self.get_parameter(
                "publish_markers"
            ).value
        )

        self._publish_det3d = (
            self.get_parameter(
                "publish_detections_3d"
            ).value
        )

        self._tf_timeout = (
            self.get_parameter(
                "tf_timeout"
            ).value
        )

        self._max_tf_staleness = (
            self.get_parameter(
                "max_tf_staleness"
            ).value
        )

        targets = self.get_parameter(
            "target_classes"
        ).value or []

        self._target_classes = {
            name
            for name in targets
            if name
        }

        # ==============================================================
        # Load YOLO
        # ==============================================================

        self._model = self._load_model()

        self._class_names = (
            self._model.names
        )

        # ==============================================================
        # Camera / image state
        # ==============================================================

        self._bridge = CvBridge()

        self._camera_model = (
            PinholeCameraModel()
        )

        self._have_camera_info = False

        # Prevent inference backlog.
        self._busy = False

        # Number of markers published by previous frame.
        self._prev_marker_count = 0

        # ==============================================================
        # TF2
        # ==============================================================

        self._tf_buffer = Buffer()

        self._tf_listener = (
            TransformListener(
                self._tf_buffer,
                self,
            )
        )

        self._tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        # ==============================================================
        # Publishers
        # ==============================================================

        self._det3d_pub = None

        if self._publish_det3d:

            self._det3d_pub = (
                self.create_publisher(
                    Detection3DArray,
                    "/hybraut/"
                    "hybraut_irb140/"
                    "ball_detections_3d",
                    qos_profile_system_default,
                )
            )

        self._marker_pub = None

        if self._publish_markers:

            self._marker_pub = (
                self.create_publisher(
                    MarkerArray,
                    "/hybraut/"
                    "hybraut_irb140/"
                    "ball_markers",
                    qos_profile_system_default,
                )
            )

        self._annotated_pub = None

        if self._publish_annotated:

            self._annotated_pub = (
                self.create_publisher(
                    Image,
                    "/hybraut/"
                    "hybraut_irb140/"
                    "ball_detections/"
                    "image",
                    qos_profile_system_default,
                )
            )

        # ==============================================================
        # CameraInfo
        # ==============================================================

        self.create_subscription(
            CameraInfo,
            self.get_parameter(
                "camera_info_topic"
            ).value,
            self._camera_info_cb,
            qos_profile_system_default,
        )

        # ==============================================================
        # RGB + depth synchronisation
        # ==============================================================

        self._rgb_sub = Subscriber(
            self,
            Image,
            self.get_parameter(
                "rgb_topic"
            ).value,
        )

        self._depth_sub = Subscriber(
            self,
            Image,
            self.get_parameter(
                "depth_topic"
            ).value,
        )

        self._sync = (
            ApproximateTimeSynchronizer(
                [
                    self._rgb_sub,
                    self._depth_sub,
                ],
                queue_size=5,
                slop=0.05,
            )
        )

        self._sync.registerCallback(
            self._image_cb
        )

        # ==============================================================
        # Startup information
        # ==============================================================

        self.get_logger().info(
            "ball_detector ready"
        )

        self.get_logger().info(
            f"YOLO classes: "
            f"{list(self._class_names.values())}"
        )

        self.get_logger().info(
            f"Target classes: "
            f"{sorted(self._target_classes) or 'all'}"
        )

        self.get_logger().info(
            f"Base frame: "
            f"{self._base_frame}"
        )

        self.get_logger().info(
            f"Optical camera frame: "
            f"{self._camera_optical_frame}"
        )

        self.get_logger().info(
            "3D detections will be transformed "
            f"from {self._camera_optical_frame} "
            f"to {self._base_frame} using TF2"
        )

    # ==================================================================
    # YOLO
    # ==================================================================

    def _load_model(self):

        from ultralytics import YOLO

        model_path = self.get_parameter(
            "model_path"
        ).value

        if not model_path:

            packaged = os.path.join(
                get_package_share_directory(
                    "hybraut_irb140"
                ),
                "weights",
                "ball_yolo.pt",
            )

            if os.path.isfile(packaged):

                model_path = packaged

            else:

                model_path = (
                    "yolov8n.pt"
                )

        self.get_logger().info(
            f"Loading YOLO weights: "
            f"{model_path} "
            f"(device={self._device})"
        )

        model = YOLO(
            model_path
        )

        try:
            model.fuse()
        except Exception:
            pass

        return model

    # ==================================================================
    # CameraInfo
    # ==================================================================

    def _camera_info_cb(
        self,
        msg: CameraInfo,
    ) -> None:

        self._camera_model.fromCameraInfo(
            msg
        )

        self._have_camera_info = True

    # ==================================================================
    # Image callback
    # ==================================================================

    def _image_cb(
        self,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> None:

        if not self._have_camera_info:
            return

        if self._busy:
            return

        self._busy = True

        try:

            self._process(
                rgb_msg,
                depth_msg,
            )

        finally:

            self._busy = False

    # ==================================================================
    # Main perception pipeline
    # ==================================================================

    def _process(
        self,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> None:

        # --------------------------------------------------------------
        # Convert RGB
        # --------------------------------------------------------------

        bgr = (
            self._bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8",
            )
        )

        # --------------------------------------------------------------
        # Convert depth to metres
        # --------------------------------------------------------------

        depth = depth_to_meters(
            self._bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        )

        # --------------------------------------------------------------
        # YOLO inference
        # --------------------------------------------------------------

        result = self._model.predict(
            bgr,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]

        # --------------------------------------------------------------
        # Extract relevant detections
        # --------------------------------------------------------------

        balls = []

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            name = self._class_names.get(
                class_id,
                str(class_id),
            )

            if (
                self._target_classes
                and name not in self._target_classes
            ):
                continue

            score = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                float(v)
                for v in box.xyxy[0].tolist()
            )

            balls.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                    name,
                )
            )

        # --------------------------------------------------------------
        # Output messages
        # --------------------------------------------------------------

        overlay = bgr.copy()

        det3d_array = (
            Detection3DArray()
        )

        det3d_array.header.frame_id = (
            self._base_frame
        )

        # Stamped from the depth image, not the RGB image: this is the
        # timestamp the ball's 3D position was actually measured at (and
        # therefore the one the base_link<->camera TF lookup in
        # `_transform_point()` uses). RGB/depth are only approximately
        # synced (`slop`), so using rgb_msg's stamp here would introduce
        # up to a `slop`-sized mismatch between the pose used to compute
        # the point and the pose implied by the published timestamp.
        det3d_array.header.stamp = (
            depth_msg.header.stamp
        )

        markers = MarkerArray()

        transforms: List[
            TransformStamped
        ] = []

        # Camera focal length.
        fx = self._camera_model.fx()

        single = (
            len(balls) == 1
        )

        published = 0

        # ==============================================================
        # Process each ball
        # ==============================================================

        for (
            x1,
            y1,
            x2,
            y2,
            score,
            name,
        ) in balls:

            # ----------------------------------------------------------
            # Bounding-box centre
            # ----------------------------------------------------------

            u = 0.5 * (
                x1 + x2
            )

            v = 0.5 * (
                y1 + y2
            )

            r_px = 0.25 * (
                (x2 - x1)
                +
                (y2 - y1)
            )

            # ----------------------------------------------------------
            # Draw bounding box
            # ----------------------------------------------------------

            cv2.rectangle(
                overlay,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            cv2.circle(
                overlay,
                (int(u), int(v)),
                3,
                (0, 255, 0),
                -1,
            )

            label = (
                f"{name} {score:.2f}"
            )

            # ----------------------------------------------------------
            # Pixel -> 3D optical coordinate
            # ----------------------------------------------------------

            pt_cam = backproject_pixel(
                u,
                v,
                depth,
                self._camera_model,
                window=self._depth_window,
                min_valid_depth_m=(
                    self._min_valid_depth_m
                ),
            )

            if (
                pt_cam is None
                or score < self._min_score_for_3d
            ):

                if pt_cam is None:

                    self.get_logger().debug(
                        f"No valid depth for "
                        f"'{name}' at "
                        f"({u:.0f}, {v:.0f})"
                    )

                self._draw_label(
                    overlay,
                    label,
                    x1,
                    y1,
                )

                continue

            # ----------------------------------------------------------
            # CRITICAL FRAME STEP
            # ----------------------------------------------------------
            #
            # `backproject_pixel()` returns:
            #
            #     x = right
            #     y = down
            #     z = forward
            #
            # Therefore this point is an OPTICAL-frame point.
            #
            # Do NOT use:
            #
            #     depth_camera
            #
            # because the Gazebo body frame is:
            #
            #     x = forward
            #     y = left
            #     z = up
            #
            # Instead use:
            #
            #     depth_camera_optical
            #
            # TF2 will then apply the fixed optical-frame rotation.
            # ----------------------------------------------------------

            pt_cam.header.frame_id = (
                self._camera_optical_frame
            )

            pt_cam.header.stamp = (
                depth_msg.header.stamp
            )

            # ----------------------------------------------------------
            # Depth to approximate sphere centre
            # ----------------------------------------------------------

            z = pt_cam.point.z

            r_m = (
                r_px * z / fx
            )

            # The depth measurement generally corresponds to the
            # visible/front surface of the ball.
            #
            # Move approximately one radius backwards along the
            # optical Z axis to estimate the sphere centre.
            pt_cam.point.z += r_m

            # ----------------------------------------------------------
            # Transform optical camera -> base_link
            # ----------------------------------------------------------

            pt_base = (
                self._transform_point(
                    pt_cam
                )
            )

            if pt_base is None:

                self._draw_label(
                    overlay,
                    label,
                    x1,
                    y1,
                )

                continue

            # ----------------------------------------------------------
            # Successful detection
            # ----------------------------------------------------------

            self._draw_label(
                overlay,
                f"{label} {z:.2f}m",
                x1,
                y1,
            )

            diameter = max(
                2.0 * r_m,
                1e-3,
            )

            # ==========================================================
            # Detection3D
            # ==========================================================

            if self._det3d_pub is not None:

                det3d = Detection3D()

                det3d.header = (
                    det3d_array.header
                )

                det3d.bbox.center.position.x = (
                    pt_base.point.x
                )

                det3d.bbox.center.position.y = (
                    pt_base.point.y
                )

                det3d.bbox.center.position.z = (
                    pt_base.point.z
                )

                det3d.bbox.center.orientation.w = (
                    1.0
                )

                det3d.bbox.size.x = (
                    diameter
                )

                det3d.bbox.size.y = (
                    diameter
                )

                det3d.bbox.size.z = (
                    diameter
                )

                hyp = (
                    ObjectHypothesisWithPose()
                )

                hyp.hypothesis.class_id = (
                    name
                )

                hyp.hypothesis.score = (
                    score
                )

                hyp.pose.pose.position.x = (
                    pt_base.point.x
                )

                hyp.pose.pose.position.y = (
                    pt_base.point.y
                )

                hyp.pose.pose.position.z = (
                    pt_base.point.z
                )

                hyp.pose.pose.orientation.w = (
                    1.0
                )

                det3d.results.append(
                    hyp
                )

                det3d_array.detections.append(
                    det3d
                )

            # ==========================================================
            # RViz sphere
            # ==========================================================

            if self._marker_pub is not None:

                marker = Marker()

                marker.header.frame_id = (
                    self._base_frame
                )

                # See det3d_array.header.stamp above: use the depth
                # capture time, not the RGB one, to match the pose the
                # TF lookup used.
                marker.header.stamp = (
                    depth_msg.header.stamp
                )

                marker.ns = (
                    self._marker_ns
                )

                marker.id = (
                    published
                )

                marker.type = (
                    Marker.SPHERE
                )

                marker.action = (
                    Marker.ADD
                )

                marker.pose.position.x = (
                    pt_base.point.x
                )

                marker.pose.position.y = (
                    pt_base.point.y
                )

                marker.pose.position.z = (
                    pt_base.point.z
                )

                marker.pose.orientation.w = (
                    1.0
                )

                marker.scale.x = (
                    diameter
                )

                marker.scale.y = (
                    diameter
                )

                marker.scale.z = (
                    diameter
                )

                marker.color.r = 0.1
                marker.color.g = 1.0
                marker.color.b = 0.1

                marker.color.a = float(
                    self._marker_alpha
                )

                marker.lifetime = (
                    Duration(
                        seconds=(
                            self._marker_lifetime
                        )
                    ).to_msg()
                )

                markers.markers.append(
                    marker
                )

            # ==========================================================
            # TF: base_link -> ball
            # ==========================================================

            child = (
                self._tf_frame
                if single
                else (
                    f"{self._tf_frame}_"
                    f"{published}"
                )
            )

            transform = (
                TransformStamped()
            )

            # See det3d_array.header.stamp above: use the depth capture
            # time, not the RGB one, to match the pose the TF lookup
            # used.
            transform.header.stamp = (
                depth_msg.header.stamp
            )

            transform.header.frame_id = (
                self._base_frame
            )

            transform.child_frame_id = (
                child
            )

            transform.transform.translation.x = (
                pt_base.point.x
            )

            transform.transform.translation.y = (
                pt_base.point.y
            )

            transform.transform.translation.z = (
                pt_base.point.z
            )

            transform.transform.rotation.w = (
                1.0
            )

            transforms.append(
                transform
            )

            published += 1

        # ==============================================================
        # Publish markers
        # ==============================================================

        if self._marker_pub is not None:

            for stale_id in range(
                published,
                self._prev_marker_count,
            ):

                gone = Marker()

                gone.header.frame_id = (
                    self._base_frame
                )

                gone.header.stamp = (
                    rgb_msg.header.stamp
                )

                gone.ns = (
                    self._marker_ns
                )

                gone.id = stale_id

                gone.action = (
                    Marker.DELETE
                )

                markers.markers.append(
                    gone
                )

            self._prev_marker_count = (
                published
            )

            self._marker_pub.publish(
                markers
            )

        # ==============================================================
        # Publish Detection3DArray
        # ==============================================================

        if self._det3d_pub is not None:

            self._det3d_pub.publish(
                det3d_array
            )

        # ==============================================================
        # Broadcast ball TF
        # ==============================================================

        if transforms:

            self._tf_broadcaster.sendTransform(
                transforms
            )

        # ==============================================================
        # Annotated image
        # ==============================================================

        if self._annotated_pub is not None:

            annotated_msg = (
                self._bridge.cv2_to_imgmsg(
                    overlay,
                    encoding="bgr8",
                )
            )

            annotated_msg.header = (
                rgb_msg.header
            )

            self._annotated_pub.publish(
                annotated_msg
            )

    # ==================================================================
    # Drawing helper
    # ==================================================================

    @staticmethod
    def _draw_label(
        image,
        text: str,
        x: float,
        y: float,
    ) -> None:

        cv2.putText(
            image,
            text,
            (
                int(x),
                max(
                    0,
                    int(y) - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ==================================================================
    # TF transform
    # ==================================================================

    def _transform_point(
        self,
        point: PointStamped,
    ) -> Optional[PointStamped]:
        """
        Transform an optical-frame point into base_frame.

        Expected transformation:

            depth_camera_optical
                    |
                    | TF2
                    v
                base_link

        The first attempt uses the image/depth timestamp because this
        is an eye-in-hand camera and its pose changes as the robot moves.

        If the exact timestamp is unavailable because simulation TF
        publication lags slightly behind the camera timestamp, the latest
        available transform is used as a fallback -- but only if it is
        within `max_tf_staleness` of the point's true capture time. Since
        the camera moves with the arm, blindly accepting an arbitrarily
        stale fallback transform introduces a position error proportional
        to how far the arm moved in the gap; a dropped detection is far
        safer for downstream motion/pick logic than a confidently wrong
        one.
        """

        capture_stamp = (
            point.header.stamp
        )

        # --------------------------------------------------------------
        # Attempt 1:
        # Exact depth timestamp
        # --------------------------------------------------------------

        try:

            return self._tf_buffer.transform(
                point,
                self._base_frame,
                timeout=Duration(
                    seconds=self._tf_timeout
                ),
            )

        except tf2_ros.ExtrapolationException:

            pass

        except tf2_ros.TransformException as e:

            self.get_logger().warning(
                f"Could not transform point "
                f"from '{point.header.frame_id}' "
                f"to '{self._base_frame}' "
                f"at image stamp: "
                f"{e}"
            )

            return None

        # --------------------------------------------------------------
        # Attempt 2:
        # Latest available transform, bounded by max_tf_staleness
        # --------------------------------------------------------------

        if self._max_tf_staleness <= 0.0:

            self.get_logger().warning(
                f"Could not transform point "
                f"from '{point.header.frame_id}' "
                f"to '{self._base_frame}': "
                f"no TF at image stamp within "
                f"{self._tf_timeout:.3f}s, and the "
                f"latest-TF fallback is disabled "
                f"(max_tf_staleness <= 0)"
            )

            return None

        point_latest = PointStamped()

        point_latest.header = (
            point.header
        )

        point_latest.header.stamp = (
            Time().to_msg()
        )

        point_latest.point = (
            point.point
        )

        try:

            pt_latest = self._tf_buffer.transform(
                point_latest,
                self._base_frame,
                timeout=Duration(
                    seconds=self._tf_timeout
                ),
            )

        except tf2_ros.TransformException as e:

            self.get_logger().warning(
                f"Could not transform point "
                f"from '{point.header.frame_id}' "
                f"to '{self._base_frame}' "
                f"at image timestamp or latest "
                f"TF: {e}"
            )

            return None

        # `do_transform_point()` stamps the result with the transform's
        # own evaluation time (not the input point's stamp), so this is
        # the actual staleness of the pose that was used -- not just how
        # long the lookup took.
        staleness = (
            abs(
                (
                    Time.from_msg(
                        pt_latest.header.stamp
                    )
                    - Time.from_msg(
                        capture_stamp
                    )
                ).nanoseconds
            )
            * 1e-9
        )

        if staleness > self._max_tf_staleness:

            self.get_logger().warning(
                f"Dropping ball detection: "
                f"latest available TF for "
                f"'{self._base_frame}' is "
                f"{staleness * 1000:.0f} ms from "
                f"the image's capture stamp "
                f"(limit "
                f"{self._max_tf_staleness * 1000:.0f} "
                f"ms) -- arm is likely moving too "
                f"fast for TF publication to keep up"
            )

            return None

        return pt_latest


def main() -> None:

    rclpy.init()

    node = BallDetector()

    # MultiThreadedExecutor (rather than plain rclpy.spin(), which is a
    # SingleThreadedExecutor) gives the TF listener's own
    # ReentrantCallbackGroup (set up internally by tf2_ros.TransformListener)
    # a real concurrency lane separate from the image-processing callback's
    # default MutuallyExclusiveCallbackGroup. Under a single thread, a busy
    # image callback can starve the TF listener's callback for multiple
    # seconds under heavy load (e.g. a real MoveIt trajectory execution),
    # leaving _transform_point()'s "latest available" fallback evaluating
    # against a stale tf2_ros.Buffer even though /tf itself is being
    # published on time.
    executor = MultiThreadedExecutor(num_threads=4)

    executor.add_node(node)

    try:

        executor.spin()

    except KeyboardInterrupt:

        pass

    finally:

        executor.shutdown()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()

