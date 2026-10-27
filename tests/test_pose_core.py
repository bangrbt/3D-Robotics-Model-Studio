import math
from copy import deepcopy
import unittest

import numpy as np

from pose_robot.core import Observation, PoseController, WaveDetector, arm_sample, joint_angle, robot_points
from pose_robot.graphics import demo_observation


class PoseGeometryTests(unittest.TestCase):
    def test_angles_and_robot_convention_match(self):
        for heading in (-179,-90,0,45,179):
            for angle in (0,30,90,135,180):
                for bend in (-1,1):
                    points=robot_points(heading,angle,bend,(15,20),(130,110))
                    self.assertAlmostEqual(joint_angle(*points),angle,places=5)

    def test_degenerate_or_non_finite_geometry_is_rejected(self):
        for points in (((0,0),(0,0),(1,0)),((0,0),(1,0),(float("nan"),0))):
            with self.assertRaises(ValueError):
                joint_angle(*points)

    def test_aspect_correction_and_mirror_identity(self):
        obs=demo_observation(0)
        obs.width,obs.height=1280,720
        for index,(x,y) in zip((12,14,16),((100,100),(200,200),(300,100))):
            obs.landmarks[index].x=x/1280
            obs.landmarks[index].y=y/720
        original=arm_sample(obs,"right",False)
        mirrored=arm_sample(obs,"right",True)
        self.assertAlmostEqual(original.angle,90)
        self.assertAlmostEqual(mirrored.angle,90)
        self.assertAlmostEqual(original.heading,45)
        self.assertAlmostEqual(mirrored.heading,135)
        self.assertEqual(mirrored.side,"right")
        self.assertEqual(original.bend,-mirrored.bend)

    def test_low_confidence_clipped_and_zero_length_arms_are_rejected(self):
        for field,value in (("visibility",.2),("presence",.2),("x",1.2),("y",float("nan"))):
            obs=demo_observation(0)
            setattr(obs.landmarks[16],field,value)
            self.assertIsNone(arm_sample(obs,"right"))
        obs=demo_observation(0)
        obs.landmarks[14]=deepcopy(obs.landmarks[12])
        self.assertIsNone(arm_sample(obs,"right"))


class PoseActionTests(unittest.TestCase):
    def feed(self,controller,pose,frames=30,start=0,**options):
        for i in range(frames):
            state=controller.update(pose,start+i/30,**options)
        return state

    def test_actions_require_five_consecutive_valid_frames(self):
        controller=PoseController()
        for i in range(4):
            state=controller.update(demo_observation(0),i/30)
            self.assertIsNone(state.event)
            self.assertEqual(state.action,"WAIT")
        self.assertEqual(controller.update(demo_observation(0),4/30).event,"RAISE")
        self.assertIsNone(controller.update(demo_observation(0),5/30).event)
        self.assertEqual(self.feed(controller,demo_observation(3),start=1).action,"LOWER")

    def test_loss_invalidates_angle_and_breaks_streak(self):
        controller=PoseController()
        self.feed(controller,demo_observation(0),4)
        state=controller.update(Observation(),.2)
        self.assertFalse(state.valid)
        self.assertIsNone(state.angle)
        self.assertIsNone(state.event)
        self.assertEqual(self.feed(controller,demo_observation(0),4,start=.3).frames,4)
        self.assertEqual(controller.update(demo_observation(0),.45).event,"RAISE")

    def test_gap_cannot_accumulate_old_frames(self):
        controller=PoseController()
        self.feed(controller,demo_observation(0),4)
        state=controller.update(demo_observation(0),2)
        self.assertEqual(state.frames,1)
        self.assertIsNone(state.event)

    def test_static_raised_arm_is_not_wave(self):
        controller=PoseController()
        for i in range(300):
            state=controller.update(demo_observation(0),i/30)
            self.assertNotEqual(state.action,"WAVE")

    def test_wave_requires_real_reversals_and_expires(self):
        controller=PoseController()
        actions=[]
        for i in range(300):
            now=i/30
            state=controller.update(demo_observation(5+now),now)
            if state.event:
                actions.append(state.event)
        self.assertIn("WAVE",actions)
        state=self.feed(controller,demo_observation(0),frames=100,start=10)
        self.assertEqual(state.action,"RAISE")

    def test_wrist_noise_and_body_translation_do_not_trigger_wave(self):
        wave=WaveDetector()
        for i in range(300):
            self.assertFalse(wave.update(.03*math.sin(i),True,i/30))
        controller=PoseController()
        for i in range(150):
            pose=demo_observation(0)
            for point in pose.landmarks:
                point.x+=.05*math.sin(i/10)
            state=controller.update(pose,i/30)
            self.assertNotEqual(state.action,"WAVE")

    def test_old_wave_history_cannot_keep_wave_active_after_motion_stops(self):
        wave=WaveDetector()
        active=False
        for i in range(90):
            value=.7*math.sin(i/30*2*math.pi*1.5)
            active=wave.update(value,True,i/30) or active
        self.assertTrue(active)
        for i in range(90,130):
            state=wave.update(value,True,i/30)
        self.assertFalse(state)

    def test_auto_selection_stays_on_same_arm_until_loss(self):
        controller=PoseController()
        pose=demo_observation(0)
        for index in (11,13,15):
            pose.landmarks[index].visibility=.8
        self.assertEqual(self.feed(controller,pose).side,"right")
        for index in (11,13,15):
            pose.landmarks[index].visibility=.99
        for index in (12,14,16):
            pose.landmarks[index].visibility=.75
        self.assertEqual(self.feed(controller,pose,start=1).side,"right")
        pose.landmarks[16].visibility=.1
        self.assertFalse(controller.update(pose,2).valid)
        self.assertEqual(self.feed(controller,pose,30,start=2.1).side,"left")

    def test_robot_elbow_is_the_displayed_filtered_angle(self):
        controller=PoseController()
        for i in range(150):
            state=controller.update(demo_observation(i/30),i/30)
            self.assertAlmostEqual(joint_angle(*robot_points(state.shoulder_heading,state.robot_elbow,state.bend)),state.angle,places=5)


if __name__=="__main__":
    unittest.main()
