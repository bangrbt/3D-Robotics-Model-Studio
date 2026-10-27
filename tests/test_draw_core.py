from types import SimpleNamespace
import math
import unittest

import numpy as np

from drawing_robot.core import DrawingEngine,HandObservation,OneEuro,Stroke,pen_gesture
from drawing_robot.kinematics import drawing_roi,map_pointer,solve
from drawing_robot.shapes import recognize_recent,recognize_shape


def path(vertices,per_edge=22):
    return [tuple((1-t)*np.array(a)+t*np.array(b)) for a,b in zip(vertices[:-1],vertices[1:]) for t in np.linspace(0,1,per_edge)]


def circle():
    return [(float(.5+.22*np.cos(t)),float(.5+.308*np.sin(t))) for t in np.linspace(0,2*np.pi,120)]


def fake_hand(pinched=True):
    p=[SimpleNamespace(x=.5,y=.6,z=0) for _ in range(21)]
    for i,(x,y) in {0:(.5,.8),5:(.4,.6),17:(.6,.6),8:(.5,.3),4:(.505,.305) if pinched else (.25,.45)}.items():
        p[i].x,p[i].y=x,y
    return HandObservation(p,[],"Right",640,480)


class DrawingGeometryTests(unittest.TestCase):
    def test_every_sheet_target_is_reachable_and_tip_matches(self):
        for u in np.linspace(0,1,25):
            for v in np.linspace(0,1,25):
                robot=solve((u,v))
                np.testing.assert_allclose(robot.uv,(u,v),atol=1e-12)
                self.assertGreater(robot.joints[1,1],0)

    def test_roi_preserves_circle_aspect_for_all_camera_shapes(self):
        for w,h in ((640,480),(1280,720),(480,640),(1920,1080)):
            x,y,rw,rh=drawing_roi(w,h)
            self.assertAlmostEqual(rw/rh,1.4)
            self.assertGreaterEqual(x,0)
            self.assertGreaterEqual(y,0)
            p=((x+.3*rw)/w,(y+.7*rh)/h)
            np.testing.assert_allclose(map_pointer(p,w,h,False),(.3,.7))
            np.testing.assert_allclose(map_pointer((1-p[0],p[1]),w,h,True),(.3,.7))

    def test_outside_targets_are_rejected_not_clamped_into_an_edge_line(self):
        self.assertIsNone(map_pointer((0,0),640,480))
        with self.assertRaises(ValueError):
            solve((1.1,.4))

    def test_pinch_detects_both_hands_and_neutral_release(self):
        self.assertTrue(pen_gesture(fake_hand(True))[0])
        self.assertFalse(pen_gesture(fake_hand(False))[0])
        self.assertIsNone(pen_gesture(HandObservation())[0])

    def test_index_mode_draws_with_one_finger_and_lifts_with_two(self):
        hand=fake_hand(False)
        for i,(x,y) in {5:(.4,.6),6:(.4,.45),8:(.4,.2),9:(.55,.6),10:(.55,.4),12:(.55,.55)}.items():
            hand.landmarks[i].x,hand.landmarks[i].y=x,y
        self.assertTrue(pen_gesture(hand,"index")[0])
        hand.landmarks[12].y=.2
        self.assertFalse(pen_gesture(hand,"index")[0])


class LiveDrawingTests(unittest.TestCase):
    def test_pen_starts_on_fifth_frame_and_keeps_buffered_start(self):
        engine=DrawingEngine()
        for i in range(4):
            engine.update_pointer((.2+i*.01,.3),True,i/30)
            self.assertFalse(engine.pen_down)
            self.assertIsNone(engine.current)
        engine.update_pointer((.24,.3),True,4/30)
        self.assertTrue(engine.pen_down)
        self.assertEqual(len(engine.current.points),5)
        self.assertAlmostEqual(engine.current.points[0][0],.2)

    def test_robot_appends_actual_pen_position_each_frame_before_shape_finishes(self):
        engine=DrawingEngine()
        for i in range(40):
            engine.update_pointer((.15+i*.012,.4),True,i/30)
            if engine.pen_down:
                np.testing.assert_allclose(engine.current.points[-1],engine.robot.uv)
        self.assertGreater(len(engine.current.points),30)
        self.assertEqual(engine.shape.key,"none")
        engine.update_pointer((.7,.4),False,40/30)
        self.assertEqual(engine.shape.key,"line")

    def test_release_is_immediate_and_no_travel_line_is_drawn(self):
        engine=DrawingEngine()
        for i in range(6):
            engine.update_pointer((.2+i*.01,.2),True,i/30)
        engine.update_pointer((.3,.4),False,.2)
        self.assertFalse(engine.pen_down)
        for i in range(8):
            engine.update_pointer((.7,.7),False,.3+i/30)
        self.assertEqual(len(engine.strokes),1)
        self.assertLess(max(p[0] for p in engine.strokes[0].points),.3)

    def test_loss_gap_jump_and_identity_swap_break_strokes(self):
        for failure in ("loss","gap","jump","swap"):
            engine=DrawingEngine()
            for i in range(6):
                engine.update_pointer((.2,.3),True,i/30,"Left")
            if failure=="loss":
                engine.update_pointer(None,True,.2,"Left")
            else:
                engine.update_pointer((.85,.85) if failure=="jump" else (.2,.3),True,
                                      2 if failure=="gap" else .2,"Right" if failure=="swap" else "Left")
            self.assertFalse(engine.pen_down)
            self.assertEqual(len(engine.strokes),1)
            self.assertIsNone(engine.current)

    def test_clear_and_undo_cannot_recreate_ink_until_release(self):
        for operation in ("clear","undo"):
            engine=DrawingEngine()
            for i in range(6):
                engine.update_pointer((.4,.4),True,i/30)
            getattr(engine,operation)()
            for i in range(15):
                engine.update_pointer((.4,.4),True,.3+i/30)
            self.assertFalse(engine.all_strokes())
            engine.update_pointer((.4,.4),False,1)
            for i in range(5):
                engine.update_pointer((.4,.4),True,1.1+i/30)
            self.assertTrue(engine.pen_down)

    def test_color_changes_preserve_existing_strokes(self):
        engine=DrawingEngine()
        for i in range(6):
            engine.update_pointer((.4,.4),True,i/30)
        engine.change_style("#EA6A42",5)
        self.assertEqual(engine.strokes[0].color,"#285CE6")
        self.assertEqual(engine.color,"#EA6A42")
        self.assertEqual(engine.width,5)

    def test_invalid_frame_interrupts_real_hand_pen_gate(self):
        engine=DrawingEngine()
        for i in range(4):
            engine.update(fake_hand(),i/30)
        engine.update(HandObservation(),4/30)
        engine.update(fake_hand(),5/30)
        self.assertEqual(engine.frames,1)
        self.assertFalse(engine.pen_down)

    def test_filter_reduces_jitter_without_large_moving_pointer_lag(self):
        filter=OneEuro()
        values=[]
        raw=[]
        for i in range(120):
            value=.5+.008*math.sin(i*2)
            raw.append(value)
            values.append(filter.update((value,.5),i/30)[0])
        self.assertLess(np.std(values[20:]),np.std(raw[20:])*.65)
        for i in range(30):
            target=.5+i*.01
            actual=filter.update((target,.5),4+i/30)[0]
        self.assertLess(target-actual,.04)


class ShapeTests(unittest.TestCase):
    def test_basic_shapes(self):
        cases={"circle":circle(),
               "square":path(((.25,.15),(.75,.15),(.75,.85),(.25,.85),(.25,.15))),
               "rectangle":path(((.15,.25),(.85,.25),(.85,.75),(.15,.75),(.15,.25))),
               "triangle":path(((.5,.15),(.8,.85),(.2,.85),(.5,.15))),
               "line":path(((.2,.2),(.8,.8))),
               "ellipse":[(.5+.3*math.cos(t),.5+.17*math.sin(t)) for t in np.linspace(0,2*math.pi,120)]}
        for expected,points in cases.items():
            self.assertEqual(recognize_shape(points).key,expected,expected)

    def test_shape_recognition_works_after_live_smoothing(self):
        engine=DrawingEngine()
        for i,uv in enumerate(circle()):
            engine.update_pointer(uv,True,i/30)
        engine.update_pointer(circle()[-1],False,4.1)
        self.assertEqual(engine.shape.key,"circle")

    def test_open_circle_and_concave_star_are_not_basic_closed_shapes(self):
        self.assertEqual(recognize_shape(circle()[:85]).key,"freehand")
        star=[]
        for i in range(11):
            t=i*math.pi/5
            r=.35 if i%2==0 else .13
            star.append((.5+r*math.cos(t)/1.4,.5+r*math.sin(t)))
        self.assertEqual(recognize_shape(path(star)).key,"freehand")

    def test_separate_connected_edges_can_form_a_square(self):
        vertices=((.25,.15),(.75,.15),(.75,.85),(.25,.85),(.25,.15))
        strokes=[Stroke(path((a,b))) for a,b in zip(vertices[:-1],vertices[1:])]
        self.assertEqual(recognize_recent(strokes).key,"square")

    def test_recognition_does_not_replace_or_snap_points(self):
        points=circle()
        original=list(points)
        recognize_shape(points)
        self.assertEqual(points,original)

    def test_rotated_rectangle_is_recognized(self):
        vertices=np.array(((-.35,-.2),(.35,-.2),(.35,.2),(-.35,.2),(-.35,-.2)))
        angle=math.radians(28)
        rotation=np.array(((math.cos(angle),-math.sin(angle)),(math.sin(angle),math.cos(angle))))
        vertices=vertices@rotation.T+np.array((.7,.5))
        vertices/=np.array((1.4,1))
        self.assertEqual(recognize_shape(path(vertices)).key,"rectangle")


if __name__=="__main__":
    unittest.main()
