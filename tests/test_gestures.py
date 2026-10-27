import unittest

from gestures import GESTURES, StableGestureGate
from scene3d import Scene3D


class GateTests(unittest.TestCase):
    def test_all_seven_emit_on_exactly_fifth_frame(self):
        for gesture,(_,command,_) in GESTURES.items():
            gate=StableGestureGate()
            for index in range(1,5):
                state=gate.update(gesture,0.9,index/30)
                self.assertEqual(state.frames,index)
                self.assertIsNone(state.event)
                self.assertFalse(state.stable)
            state=gate.update(gesture,0.9,5/30)
            self.assertEqual(state.event,command)
            self.assertTrue(state.stable)

    def test_hold_does_not_repeat(self):
        gate=StableGestureGate()
        events=[gate.update("Victory",0.95,i/30).event for i in range(100)]
        self.assertEqual([e for e in events if e],["CHANGE_COLOR"])

    def test_any_bad_frame_breaks_streak(self):
        for bad,confidence in ((None,0),("None",0.95),("Victory",0.64),("Victory",float("nan"))):
            gate=StableGestureGate()
            for i in range(4):
                gate.update("Victory",0.95,i/30)
            self.assertEqual(gate.update(bad,confidence,4/30).frames,0)
            for i in range(4):
                self.assertIsNone(gate.update("Victory",0.95,(5+i)/30).event)
            self.assertEqual(gate.update("Victory",0.95,9/30).event,"CHANGE_COLOR")

    def test_alternating_gestures_never_emit(self):
        gate=StableGestureGate()
        for i in range(100):
            self.assertIsNone(gate.update("Victory" if i%2 else "Open_Palm",0.9,i/30).event)

    def test_rearm_needs_five_neutral_frames(self):
        gate=StableGestureGate()
        for _ in range(5):
            gate.update("Victory",0.9)
        for _ in range(4):
            gate.update(None,0)
        for _ in range(8):
            self.assertIsNone(gate.update("Victory",0.9).event)
        for _ in range(5):
            gate.update(None,0)
        for _ in range(4):
            self.assertIsNone(gate.update("Victory",0.9).event)
        self.assertEqual(gate.update("Victory",0.9).event,"CHANGE_COLOR")

    def test_switch_and_back_each_needs_five(self):
        gate=StableGestureGate()
        for gesture in ("Open_Palm","Closed_Fist","Open_Palm"):
            for _ in range(4):
                self.assertIsNone(gate.update(gesture,0.9).event)
            self.assertEqual(gate.update(gesture,0.9).event,GESTURES[gesture][1])

    def test_no_streak_across_hand_swap_or_capture_gap(self):
        gate=StableGestureGate()
        for i in range(4):
            gate.update("Thumb_Up",0.9,i/30,"Left")
        self.assertEqual(gate.update("Thumb_Up",0.9,4/30,"Right").frames,1)
        self.assertEqual(gate.update("Thumb_Up",0.9,2,"Right").frames,1)
        self.assertEqual(gate.update("Thumb_Up",0.9,2,"Right").frames,1)

    def test_custom_threshold_must_still_be_at_least_five(self):
        with self.assertRaises(ValueError):
            StableGestureGate(frames=4)
        gate=StableGestureGate(frames=8)
        for _ in range(7):
            self.assertIsNone(gate.update("Open_Palm",0.9).event)
        self.assertEqual(gate.update("Open_Palm",0.9).event,"START")

    def test_scene_changes_only_after_valid_command(self):
        gate=StableGestureGate()
        scene=Scene3D()
        for _ in range(4):
            state=gate.update("Open_Palm",0.9)
            if state.event:
                scene.apply(state.event)
        self.assertFalse(scene.running)
        scene.apply(gate.update("Open_Palm",0.9).event)
        self.assertTrue(scene.running)
        scene.apply("STOP")
        self.assertFalse(scene.running)
        for _ in range(50):
            scene.apply("ZOOM_IN")
        self.assertEqual(scene.scale,1.5)
        for _ in range(50):
            scene.apply("ZOOM_OUT")
        self.assertEqual(scene.scale,0.45)
        scene.apply("PARTICLES")
        self.assertTrue(scene.particles)


if __name__=="__main__":
    unittest.main()
