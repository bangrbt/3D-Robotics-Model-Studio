from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest
import cv2
import numpy as np

from compare import transform
from experiments import ExperimentLab
from gestures import StableGestureGate
from scene3d import Scene3D
from palm_ar import PalmAR
from ui import render
from vision import MODEL, Observation, create_recognizer, recognize


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(MODEL.is_file(),"Download gesture model with app.py --download-model")
    def test_blank_real_inference_never_emits(self):
        gate=StableGestureGate()
        with create_recognizer() as recognizer:
            for i in range(6):
                observation=recognize(recognizer,np.zeros((480,640,3),np.uint8),i*33)
                self.assertFalse(observation.landmarks)
                self.assertIsNone(gate.update(observation.gesture,observation.confidence).event)

    def test_all_commands_render_in_3d_with_particles(self):
        with TemporaryDirectory() as directory:
            lab=ExperimentLab(Path(directory)/"trials.csv")
            scene=Scene3D()
            for command in ("START","STOP","MOVE_UP","ZOOM_IN","ZOOM_OUT","CHANGE_COLOR","PARTICLES"):
                scene.apply(command)
                scene.update(1/30)
                state=StableGestureGate().update(None,0)
                output=render(np.zeros((360,640,3),np.uint8),Observation(),state,scene,lab,0,
                              last_command=(command,command),event_age=0)
                self.assertEqual(output.shape,(960,1280,3))

    def test_digital_transforms_preserve_frame_dimensions(self):
        from experiments import CONDITIONS
        frame=np.full((480,640,3),200,np.uint8)
        for condition in CONDITIONS:
            output=transform(frame,condition)
            self.assertEqual(output.shape,frame.shape)
            self.assertEqual(output.dtype,np.uint8)
        self.assertLess(transform(frame,"low_light").mean(),frame.mean())

    def test_ui_composites_palm_ar_during_hold_charge_and_burst(self):
        from types import SimpleNamespace
        from gestures import GateState
        points=[SimpleNamespace(x=.35+(i%5)*.04,y=.5+(i//5)*.035,z=0) for i in range(21)]
        points[0]=SimpleNamespace(x=.45,y=.8,z=0)
        observation=Observation("Open_Palm",.95,points,"Left",100,.1)
        ar,scene=PalmAR(),Scene3D()
        with TemporaryDirectory() as directory:
            lab=ExperimentLab(Path(directory)/"trials.csv")
            for gesture,command in (("Open_Palm","START"),("Closed_Fist","STOP"),("Open_Palm","START")):
                state=GateState(gesture,5,5,True,command)
                ar.update(points,state,1/30,"Left",scene)
                canvas=render(np.full((480,640,3),100,np.uint8),observation,state,scene,lab,0,ar=ar)
                self.assertEqual(canvas.shape,(960,1280,3))
                camera=canvas[136:616,32:672]
                self.assertGreater(np.count_nonzero(camera!=100),100)

    @unittest.skipUnless(MODEL.is_file(),"Gesture model needed for video integration")
    def test_video_cli_processes_distinct_negative_frames_without_commands(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)
            video=path/"negative.avi"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30,(160,120))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(12):
                    writer.write(np.full((120,160,3),index*3,np.uint8))
            finally:
                writer.release()
            completed=subprocess.run([sys.executable,str(MODEL.parent.parent/"app.py"),"--video",str(video),
                                      "--headless","--results",str(path/"results")],capture_output=True,text=True,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertFalse((path/"results"/"commands.csv").exists())


if __name__=="__main__":
    unittest.main()
