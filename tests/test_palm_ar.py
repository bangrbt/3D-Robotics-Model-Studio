from types import SimpleNamespace
import unittest

import numpy as np

from gestures import StableGestureGate
from palm_ar import PalmAR
from scene3d import Scene3D


def hand(dx=0.0, scale=1.0):
    points=[(.50,.80),(.40,.75),(.33,.66),(.27,.59),(.22,.53),
            (.38,.55),(.36,.43),(.35,.33),(.34,.24),
            (.48,.52),(.48,.38),(.48,.27),(.48,.17),
            (.57,.55),(.59,.43),(.60,.34),(.61,.25),
            (.65,.60),(.70,.51),(.73,.44),(.76,.37)]
    return [SimpleNamespace(x=.5+(x-.5)*scale+dx,y=.65+(y-.65)*scale,z=0) for x,y in points]


class PalmARTests(unittest.TestCase):
    def setUp(self):
        self.ar=PalmAR()
        self.gate=StableGestureGate()
        self.scene=Scene3D()
        self.time=0.0

    def feed(self,name,frames=5,points=None,confidence=.95,identity="Left"):
        for _ in range(frames):
            self.time+=1/30
            state=self.gate.update(name,confidence,self.time,identity)
            if state.event:
                self.scene.apply(state.event)
            self.ar.update(hand() if points is None else points,state,1/30,identity,self.scene)
        return state

    def test_cube_waits_for_five_open_frames(self):
        self.feed("Open_Palm",4)
        self.assertFalse(self.ar.spawned)
        self.feed("Open_Palm",1)
        self.assertTrue(self.ar.spawned)
        self.assertEqual(self.ar.phase,"holding")

    def test_charge_and_release_each_need_five_frames(self):
        self.feed("Open_Palm")
        self.feed("Closed_Fist",4)
        self.assertFalse(self.ar.armed)
        self.feed("Closed_Fist",1)
        self.assertTrue(self.ar.armed)
        self.feed("Open_Palm",4)
        self.assertEqual(self.ar.burst_count,0)
        self.feed("Open_Palm",1)
        self.assertEqual(self.ar.phase,"burst")
        self.assertEqual(self.ar.burst_count,1)
        self.assertEqual(len(self.ar.positions),140)
        self.feed("Open_Palm",120)
        self.assertEqual(self.ar.phase,"holding")
        self.assertEqual(self.ar.burst_count,1)
        self.assertEqual(len(self.ar.positions),0)

    def test_low_confidence_breaks_action_streak(self):
        self.feed("Open_Palm")
        self.feed("Closed_Fist",4)
        self.feed("Closed_Fist",1,confidence=.2)
        self.feed("Closed_Fist",4)
        self.assertFalse(self.ar.armed)
        self.feed("Closed_Fist",1)
        self.assertTrue(self.ar.armed)

    def test_direct_burst_requires_existing_cube_and_stable_gesture(self):
        self.feed("ILoveYou")
        self.assertEqual(self.ar.burst_count,0)
        self.feed("Open_Palm")
        self.feed("ILoveYou",4)
        self.assertEqual(self.ar.burst_count,0)
        self.feed("ILoveYou",1)
        self.assertEqual(self.ar.burst_count,1)

    def test_hand_loss_clears_charge_and_never_causes_phantom_burst(self):
        self.feed("Open_Palm")
        self.feed("Closed_Fist")
        self.feed(None,1,points=[],confidence=0)
        self.assertFalse(self.ar.visible)
        self.assertFalse(self.ar.armed)
        self.feed("Open_Palm")
        self.assertEqual(self.ar.phase,"holding")
        self.assertEqual(self.ar.burst_count,0)

    def test_hand_swap_discards_charge(self):
        self.feed("Open_Palm")
        self.feed("Closed_Fist")
        self.feed("Open_Palm",identity="Right")
        self.assertEqual(self.ar.burst_count,0)
        self.assertFalse(self.ar.armed)

    def test_particles_stay_at_release_position_while_hand_moves(self):
        self.feed("Open_Palm")
        self.feed("Closed_Fist")
        self.feed("Open_Palm")
        origin=self.ar.burst_anchor.copy()
        self.feed("Open_Palm",15,points=hand(dx=.15))
        np.testing.assert_allclose(self.ar.burst_anchor,origin)
        self.assertGreater(self.ar.anchor[0],origin[0]+.1)

    def test_tracking_and_scale_change_with_hand(self):
        self.feed("Open_Palm",points=hand(scale=.65))
        _,old_size=self.ar._placement(self.ar.anchor,self.ar.span,self.ar.length,(0,0,640,480))
        self.feed("Open_Palm",20,points=hand(dx=.1,scale=.85))
        center,new_size=self.ar._placement(self.ar.anchor,self.ar.span,self.ar.length,(0,0,640,480))
        self.assertGreater(new_size,old_size*1.2)
        self.assertGreater(center[0],350)

    def test_overlay_is_clipped_and_blank_without_hand(self):
        frame=np.full((480,640,3),100,np.uint8)
        np.testing.assert_array_equal(self.ar.overlay(frame,(0,60,640,360),self.scene),frame)
        self.feed("Open_Palm",15)
        for name in ("Open_Palm","Closed_Fist","Open_Palm"):
            self.feed(name)
            overlay=self.ar.overlay(frame,(0,60,640,360),self.scene)
            np.testing.assert_array_equal(overlay[:60],frame[:60])
            np.testing.assert_array_equal(overlay[420:],frame[420:])
            self.assertGreater(np.count_nonzero(overlay!=frame),500)

    def test_invalid_landmarks_hide_ar(self):
        self.feed("Open_Palm")
        points=hand()
        points[5].x=float("nan")
        self.feed("Open_Palm",1,points=points)
        self.assertFalse(self.ar.visible)


if __name__=="__main__":
    unittest.main()
