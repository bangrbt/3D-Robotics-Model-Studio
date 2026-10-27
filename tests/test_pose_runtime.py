from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import subprocess
import sys
import time
import tkinter as tk
import unittest

import cv2
import numpy as np

from pose_robot.core import MotionState, Observation, PoseController
from pose_robot.graphics import camera_image, demo_frame, demo_observation, robot_image
from pose_robot.vision import MODEL, create_detector, detect
from pose_robot.worker import PoseWorker


class PoseRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(MODEL.is_file(),"Download the pose model first")
    def test_real_pose_model_rejects_empty_frames(self):
        with create_detector() as model:
            for i in range(3):
                observation=detect(model,np.zeros((480,640,3),np.uint8),i*33)
                self.assertFalse(observation.landmarks)
                state=PoseController().update(observation,i/30)
                self.assertFalse(state.valid)
                self.assertIsNone(state.event)

    @unittest.skipUnless((MODEL.parent.parent/"artifacts"/"pose_sample.jpg").is_file(),"Optional official pose image is absent")
    def test_real_sample_produces_an_arm_without_faking_five_frames(self):
        image=cv2.imread(str(MODEL.parent.parent/"artifacts"/"pose_sample.jpg"))
        with create_detector() as model:
            observation=detect(model,image,0)
        self.assertEqual(len(observation.landmarks),33)
        state=PoseController().update(observation,0)
        self.assertTrue(state.valid)
        self.assertEqual(state.frames,1)
        self.assertIsNone(state.event)
        self.assertTrue(0<=state.angle<=180)

    def test_worker_demo_stops_cleanly_and_keeps_latest_frame(self):
        worker=PoseWorker("demo")
        worker.start()
        time.sleep(.26)
        self.assertTrue(worker.thread.is_alive())
        self.assertEqual(worker.frames.qsize(),1)
        packet=worker.latest()
        self.assertIsNotNone(packet)
        self.assertTrue(packet.state.valid)
        worker.pause()
        time.sleep(.08)
        worker.stop()
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())

    def test_camera_error_is_queued_and_detector_closed(self):
        model=Mock()
        with patch("pose_robot.worker.ensure_model",return_value=MODEL),patch("pose_robot.worker.create_detector",return_value=model),patch("pose_robot.worker.open_camera",side_effect=RuntimeError("camera unavailable")):
            worker=PoseWorker("camera")
            worker.start()
            worker.thread.join(2)
        messages=[]
        while not worker.messages.empty():
            messages.append(worker.messages.get_nowait())
        self.assertIn(("error","camera unavailable"),messages)
        self.assertIn(("ended","camera"),messages)
        model.close.assert_called_once()

    def test_white_graphics_resize_without_mutating_frame(self):
        observation=demo_observation(0)
        frame=demo_frame(observation)
        state=PoseController().update(observation,0)
        original=frame.copy()
        for size in ((680,410),(480,360),(800,300),(40,40),(100,80)):
            self.assertEqual(camera_image(frame,observation,state,size).size,size)
            robot=robot_image(state,size)
            self.assertEqual(robot.size,size)
            self.assertEqual(robot.getpixel((0,0)),(255,255,255))
        np.testing.assert_array_equal(frame,original)

    @unittest.skipUnless(MODEL.is_file(),"Pose model needed")
    def test_video_cli_finishes_on_eof_without_actions_for_blank_video(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"blank.avi"
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),30,(160,120))
            self.assertTrue(writer.isOpened())
            for i in range(8):
                writer.write(np.full((120,160,3),i*3,np.uint8))
            writer.release()
            result=subprocess.run([sys.executable,str(MODEL.parent.parent/"pose_app.py"),"--video",str(path),"--headless"],capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('"event"',result.stdout)
            self.assertIn('"valid": false',result.stdout)


class NativeDesktopTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root=tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.callback_errors=[]
        self.root.report_callback_exception=lambda *error:self.callback_errors.append(error)

    def tearDown(self):
        if hasattr(self,"root"):
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self.assertFalse(getattr(self,"callback_errors",[]),"Unexpected Tk callback exception")

    def test_native_controls_and_stale_data_invalidation(self):
        from pose_robot.desktop import PoseDesktop
        app=PoseDesktop(self.root)
        self.root.update_idletasks()
        self.assertEqual(self.root.cget("bg"),"white")
        self.assertEqual(str(app.pause_button.cget("state")),"disabled")
        app.state=PoseController().update(demo_observation(0),0)
        app.invalidate()
        self.assertFalse(app.state.valid)
        self.assertIsNone(app.state.angle)
        self.assertEqual(app.state.frames,0)
        app._paint()
        self.assertEqual(app.metric_values[0].cget("text"),"—")
        self.assertEqual(app.metric_values[3].cget("text"),"Chờ tư thế")

    def test_demo_pause_resume_stop_does_not_block_tk(self):
        from pose_robot.desktop import PoseDesktop
        app=PoseDesktop(self.root)
        app.start("demo",0)
        deadline=time.monotonic()+.4
        while time.monotonic()<deadline:
            self.root.update()
            time.sleep(.01)
        self.assertIsNotNone(app.packet)
        app.pause()
        self.assertTrue(app.paused)
        app.pause()
        self.assertFalse(app.paused)
        app.stop()
        deadline=time.monotonic()+2
        while app.worker.thread.is_alive() and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(app.worker.thread.is_alive())


if __name__=="__main__":
    unittest.main()
