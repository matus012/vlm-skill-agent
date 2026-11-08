"""Phase 1 photo script (sim/INTERFACES.md section 4): four photos at absolute headings, RGB + segmentation + GT pose.

    python3 probes/photos.py --place <name> --out <dir> [--headings 0,90,180,270] [--yaw0 DEG] [--frame map|odom] [--world clinic]

`--yaw0`: absolute start heading (degrees, localization frame) instead of the yaw read at start, so the photos of a
place are at place-yaw + k*90 deg however the robot arrived (task 002, places.yaml). `<k>_seg_full.png` = the same
fused label frame uncropped (G7 negative control only; the VLM never sees it).

Run with the sim up and `source sim/env.sh` (same ROS_DOMAIN_ID / GZ_PARTITION as the sim). `--frame odom` is for the
map-less test world (sim/launch_test_seg.sh); the clinic runs Nav2/AMCL, so `map` is the default.

Frames
- RGB: `/oakd/rgb/preview/image_raw` (rgb8), depth `/oakd/rgb/preview/depth` (32FC1), same `rgbd_camera` sensor.
- Segmentation: gz topic `/oakd/seg/labels_map` (sensor `seg_camera`, semantic, co-located with `rgbd_camera`, same
  HFOV / size / clip; ros_gz_bridge started by this script). Encoding rgb8 with **the label id in every channel**
  (R == G == B == label; measured 2026-11-04, all three channels identical). This script reads channel 0.
- Self-view: the robot and anything nearer than the 0.3 m near clip are *transparent* to the segmentation camera
  (it renders what is behind them, e.g. floor 241), while the RGB camera shows them as flat background colour with
  depth -inf. So the saved `<k>_seg.png` is the segmentation label **with every pixel the RGB frame cannot see
  (depth not finite) set to 0** -- label 0 then means "background or the robot itself" as INTERFACES section 1 says,
  for the frame the VLM gets. photos.json records how many pixels that rule zeroed.

Turning: own closed-loop yaw controller on `/cmd_vel` (TwistStamped) from the localization yaw (`<frame>`->`base_link`
TF), not Nav2's spin: spin is a *relative* turn whose end is judged by odometry, while the target here is an
absolute heading in the localization frame, and a P loop on that error does not accumulate error across photos.
Settle: |v| < 0.005 m/s and |w| < 0.01 rad/s on /odom continuously for >= 0.5 s sim time, then (map frame) AMCL is
asked for 12 no-motion laser updates (`/request_nomotion_update`; AMCL otherwise updates only every 0.2 rad of motion,
and odometry over-reports rotation ~6 % in this sim), then the yaw error is re-checked (re-turn if > 0.3 deg, at most 3 rounds), then up to 3 final turns without relocalizing
while it is still > 0.3 deg (`face`, task 006 M2 round 2); frames are taken with sensor stamps after the last settle.

Ground-truth camera pose (world frame): Gazebo `/world/<w>/pose/info` gives the robot model's world pose and the
`oakd_rgb_camera_frame` link pose relative to the model; camera = model (+) link. The script checks that link offset
against the URDF TF chain `base_link`->`oakd_rgb_camera_frame` (model frame == base_link, base_footprint is an identity
child) to 1 mm / 0.1 deg and records both. GT yaw error = wrap(camera yaw_world - target), with target expressed in
the localization frame and localization frame == world (map == world is gate G8; odom == world in the test world,
where the robot spawns at the world origin with yaw 0). The localization-frame error is recorded too.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

import gz_cli

ROOT = Path(__file__).resolve().parents[1]
ROBOT_CFG = ROOT / "config/robots/turtlebot4.yaml"
MODEL, CAM_LINK = "turtlebot4", "oakd_rgb_camera_frame"
RGB_T, DEPTH_T, SEG_T = "/oakd/rgb/preview/image_raw", "/oakd/rgb/preview/depth", "/oakd/seg/labels_map"
LABEL_CHANNEL = 0  # R == G == B == label id
# controller / settle constants (instrument tuning, not gate thresholds)
KP, W_MAX, W_MIN = 1.5, 0.6, 0.04          # rad/s per rad, rad/s, rad/s
TOL_STOP = math.radians(0.15)               # stop commanding inside this error
TOL_RETURN = math.radians(0.3)              # after settling, re-turn if the error is above this
SETTLE_V, SETTLE_W, SETTLE_S = 0.005, 0.01, 0.5
NOMOTION_UPDATES = 12
FINAL_STEPS = 3  # face(): final turns without relocalizing until the residual <= TOL_RETURN (006 M2 round 2)

Pose3 = tuple[np.ndarray, np.ndarray]  # position xyz, quaternion xyzw


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return qmul(qmul(q, np.array([*v, 0.0])), q * np.array([-1, -1, -1, 1]))[:3]


def compose(a: Pose3, b: Pose3) -> Pose3:
    return a[0] + qrot(a[1], b[0]), qmul(a[1], b[1])


def gz_poses(world: str) -> tuple[dict[str, Pose3], float]:
    """Model + link poses from Gazebo pose/info (models in world frame, links relative to their model) and sim time."""
    out = gz_cli.echo_once(f"/world/{world}/pose/info", 20)  # signal deaths retried (a recorded deviation)
    msg = json.loads(out.splitlines()[0])
    poses = {}
    for p in msg["pose"]:
        pos, q = p.get("position", {}), p.get("orientation", {})
        poses.setdefault(p.get("name", ""), (np.array([pos.get(k, 0.0) for k in "xyz"]),
                                             np.array([q.get(k, 0.0) for k in "xyz"] + [q.get("w", 1.0)])))
    st = msg["header"]["stamp"]
    return poses, int(st.get("sec", 0)) + int(st.get("nsec", 0)) * 1e-9


def gt_camera(world: str) -> tuple[Pose3, Pose3, float]:
    """(camera world pose, camera offset in the model frame, sim time)."""
    poses, t = gz_poses(world)
    return compose(poses[MODEL], poses[CAM_LINK]), poses[CAM_LINK], t


def crop(img: np.ndarray, top: int, bottom: int) -> np.ndarray:
    return img[top: img.shape[0] - bottom]


def to_array(m: Image) -> np.ndarray:
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.step // 4)[:, : m.width]
    assert m.encoding == "rgb8", m.encoding
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3].reshape(m.height, m.width, 3)


def stamp(m: Image | Odometry) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def fused_labels(seg_rgb8: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, int]:
    """Label image as the RGB frame sees it: label 0 where the rgbd camera has no depth (robot / near clip)."""
    lab = seg_rgb8[..., LABEL_CHANNEL].copy()
    hidden = ~np.isfinite(depth)
    n = int((hidden & (lab != 0)).sum())
    lab[hidden] = 0
    return lab, n


class Camera(Node):
    def __init__(self, frame: str) -> None:
        super().__init__("photos115", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.frame = frame
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.last: dict[str, Image | Odometry] = {}
        for key, topic in [("rgb", RGB_T), ("depth", DEPTH_T), ("seg", SEG_T)]:
            self.create_subscription(Image, topic, lambda m, k=key: self.last.__setitem__(k, m), 5)
        self.create_subscription(Odometry, "/odom", lambda m: self.last.__setitem__("odom", m), qos_profile_sensor_data)
        self.cmd = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.nomotion = self.create_client(Empty, "/request_nomotion_update") if frame == "map" else None

    def relocalize(self) -> None:
        """map frame only: AMCL updates only after update_min_a (0.2 rad) of motion, so the end of a turn is pure odometry
        (which over-reports rotation ~6 % in this sim, 002-C). Force NOMOTION_UPDATES laser updates while standing still."""
        if self.nomotion is None:
            return
        assert self.nomotion.wait_for_service(timeout_sec=10.0), "no /request_nomotion_update (AMCL not running?)"
        for _ in range(NOMOTION_UPDATES):
            fut = self.nomotion.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
            t = self.now_s()
            self.spin_until(lambda: self.now_s() - t >= 0.3, 10.0, "AMCL update")  # >= 1 scan at 5+ Hz

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_until(self, cond, timeout_wall_s: float, what: str) -> None:  # noqa: ANN001
        end = time.monotonic() + timeout_wall_s
        while not cond():
            assert time.monotonic() < end, f"timeout waiting for {what}"
            rclpy.spin_once(self, timeout_sec=0.05)

    def loc_yaw(self) -> float:
        t = self.tf.lookup_transform(self.frame, "base_link", Time())
        q = t.transform.rotation
        return yaw_of(np.array([q.x, q.y, q.z, q.w]))

    def tf_offset(self, parent: str, child: str) -> Pose3:
        t = self.tf.lookup_transform(parent, child, Time()).transform
        return (np.array([t.translation.x, t.translation.y, t.translation.z]),
                np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]))

    def send(self, v: float, w: float) -> None:
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x, m.twist.angular.z = v, w
        self.cmd.publish(m)

    def turn_to(self, target: float, timeout_wall_s: float = 60.0) -> None:
        end = time.monotonic() + timeout_wall_s
        while True:
            err = wrap(target - self.loc_yaw())
            if abs(err) < TOL_STOP:
                break
            assert time.monotonic() < end, f"turn timeout, error {math.degrees(err):.2f} deg"
            self.send(0.0, math.copysign(min(W_MAX, max(W_MIN, KP * abs(err))), err))
            rclpy.spin_once(self, timeout_sec=0.03)
        for _ in range(5):
            self.send(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def settle(self) -> float:
        """Wait until /odom says still for SETTLE_S sim seconds; return the sim time at which it was settled."""
        since: list[float | None] = [None]

        def still() -> bool:
            o = self.last.get("odom")
            if o is None:
                return False
            tw = o.twist.twist
            if abs(tw.linear.x) < SETTLE_V and abs(tw.angular.z) < SETTLE_W:
                since[0] = since[0] if since[0] is not None else stamp(o)
                return stamp(o) - since[0] >= SETTLE_S
            since[0] = None
            return False

        self.spin_until(still, 30.0, "settle")
        return stamp(self.last["odom"])

    def frames_after(self, t: float) -> dict[str, Image]:
        self.spin_until(lambda: all(k in self.last and stamp(self.last[k]) > t for k in ("rgb", "depth", "seg")),
                        30.0, f"frames after t={t:.2f}")
        return {k: self.last[k] for k in ("rgb", "depth", "seg")}


def face(node: Camera, target: float, rounds: int) -> dict[str, float | int]:
    """Turn to an absolute localization-frame heading. Up to `rounds` x (turn_to, settle, AMCL no-motion relocalize,
    re-check <= TOL_RETURN); then (task 006 M2 round 2, 2026-11-05) up to FINAL_STEPS final turn_to + settle WITHOUT
    relocalizing while the control residual |target - loc yaw| > TOL_RETURN. Before, the loop ended on the last
    relocalize, whose estimate jump stayed in the photo (residual max 1.48 deg local run 002, 0.99 deg PERUN 143786). A
    turn of < 1 deg moves AMCL's estimate by odometry only (update_min_a 0.1 rad), so the last relocalized estimate is
    kept; the AMCL error itself is untouched. Same sim-time loops (turn_to on TF, settle on /odom stamps).
    Returns {rounds, final_steps, settled_sim_s (last settle), residual_rad (loc yaw - target)}."""
    r = f = 0
    while True:
        node.turn_to(target)
        t_settled = node.settle()
        node.relocalize()
        r += 1
        if abs(wrap(target - node.loc_yaw())) <= TOL_RETURN or r >= rounds:
            break
    while abs(wrap(target - node.loc_yaw())) > TOL_RETURN and f < FINAL_STEPS:
        node.turn_to(target)
        t_settled = node.settle()
        f += 1
    return {"rounds": r, "final_steps": f, "settled_sim_s": t_settled, "residual_rad": wrap(node.loc_yaw() - target)}


def start_seg_bridge() -> subprocess.Popen:
    return subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                             f"{SEG_T}@sensor_msgs/msg/Image[gz.msgs.Image"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def check_offset(node: Camera, gz_offset: Pose3) -> dict[str, object]:
    tf_off = node.tf_offset("base_link", CAM_LINK)
    dp = float(np.linalg.norm(tf_off[0] - gz_offset[0]))
    dq = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(tf_off[1], gz_offset[1]))))))
    assert dp < 1e-3 and dq < 0.1, f"gz camera link offset != URDF TF offset ({dp:.4f} m, {dq:.3f} deg)"
    return {"gz_model_to_cam_xyz": gz_offset[0].round(5).tolist(), "tf_base_link_to_cam_xyz": tf_off[0].round(5).tolist(),
            "diff_m": dp, "diff_deg": dq}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--headings", default="0,90,180,270", help="degrees, relative to the start yaw")
    ap.add_argument("--yaw0", type=float, default=None, help="absolute start heading, degrees (default: current yaw)")
    ap.add_argument("--frame", default="map", choices=["map", "odom"])
    ap.add_argument("--world", default="clinic")
    args = ap.parse_args()
    cfg = yaml.safe_load(ROBOT_CFG.read_text())
    top, bottom = cfg["self_crop"]["top_px"], cfg["self_crop"]["bottom_px"]
    out = args.out / args.place
    out.mkdir(parents=True, exist_ok=True)

    bridge = start_seg_bridge()
    rclpy.init()
    node = Camera(args.frame)
    try:
        node.spin_until(lambda: node.tf.can_transform(args.frame, "base_link", Time())
                        and node.tf.can_transform("base_link", CAM_LINK, Time()), 60.0, "TF")
        node.frames_after(0.0)
        start_yaw = node.loc_yaw() if args.yaw0 is None else wrap(math.radians(args.yaw0))
        _, gz_off, _ = gt_camera(args.world)
        offset_proof = check_offset(node, gz_off)
        photos = []
        for k, h in enumerate(float(x) for x in args.headings.split(",")):
            target = wrap(start_yaw + math.radians(h))
            fc = face(node, target, 3)
            rounds, t_settled = fc["rounds"], fc["settled_sim_s"]
            fr = node.frames_after(t_settled)
            cam, _, t_gt = gt_camera(args.world)
            loc_err = wrap(node.loc_yaw() - target)
            rgb = to_array(fr["rgb"])
            lab, n_hidden = fused_labels(to_array(fr["seg"]), to_array(fr["depth"]))
            rgb_c, lab_c = crop(rgb, top, bottom), crop(lab, top, bottom)
            cv2.imwrite(str(out / f"{k}_rgb.png"), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / f"{k}_seg.png"), lab_c)  # 8-bit single channel PNG = label ids, lossless
            cv2.imwrite(str(out / f"{k}_seg_full.png"), lab)  # uncropped, G7 negative control only
            yaw_w = yaw_of(cam[1])
            photos.append({
                "k": k, "heading_rel_deg": h, "target_yaw_rad": target, "target_yaw_deg": math.degrees(target),
                "gt_camera_pose_world": {"xyz": cam[0].tolist(), "quat_xyzw": cam[1].tolist(), "yaw_rad": yaw_w},
                "gt_yaw_error_deg": math.degrees(wrap(yaw_w - target)),
                "loc_yaw_error_deg": math.degrees(loc_err),
                "turn_rounds": rounds, "final_steps": fc["final_steps"], "settled_sim_s": t_settled, "gt_sim_s": t_gt,
                "frame_stamps_sim_s": {kk: stamp(m) for kk, m in fr.items()},
                "seg_px_zeroed_by_depth": n_hidden,
                "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png", "seg_full": f"{k}_seg_full.png",
            })
            print(f"photo {k}: target {math.degrees(target):8.2f} deg  GT err {photos[-1]['gt_yaw_error_deg']:+.3f} deg"
                  f"  loc err {math.degrees(loc_err):+.3f} deg  rounds {rounds}", flush=True)
        meta = {"place": args.place, "world": args.world, "loc_frame": args.frame,
                "frame_assumption": f"{args.frame} == world (map: gate G8; odom: test world spawn at world origin yaw 0)",
                "start_yaw_rad": start_yaw, "crop": {"top_px": top, "bottom_px": bottom, "rows": [top, 480 - bottom],
                                                     "source": str(ROBOT_CFG.relative_to(ROOT))},
                "label_rule": "seg label (channel 0 of labels_map); 0 where rgbd depth is not finite (robot / near clip)",
                "camera_offset_proof": offset_proof, "photos": photos}
        (out / "photos.json").write_text(json.dumps(meta, indent=2))
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(bridge.pid, signal.SIGTERM)  # ros2 run + its parameter_bridge child
        bridge.wait(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
