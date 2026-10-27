from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments import ExperimentLab, TrialMetrics, summarize
from gestures import StableGestureGate
from vision import Observation


class ExperimentTests(unittest.TestCase):
    def test_countdown_gate_reset_metrics_and_csv_reload(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"trials.csv"
            lab=ExperimentLab(path)
            gate=StableGestureGate()
            for _ in range(10):
                gate.update("Open_Palm",0.9)
            lab.start(10)
            self.assertFalse(lab.tick(11))
            self.assertTrue(lab.tick(12))
            gate.reset()
            observation=Observation("Open_Palm",0.9,[object()],"Left",120,0.1)
            for i in range(120):
                now=12+i/30
                lab.tick(now)
                state=gate.update("Open_Palm",0.9,now)
                lab.observe(observation,state,now)
            lab.tick(16)
            self.assertIsNone(lab.active)
            self.assertEqual(len(lab.rows),1)
            row=lab.rows[0]
            self.assertEqual(row["frames"],120)
            self.assertEqual(row["correct_events"],1)
            self.assertEqual(row["success"],1)
            self.assertAlmostEqual(row["first_correct_seconds"],4/30,places=3)
            summary=summarize(ExperimentLab(path).rows,"Open_Palm")
            self.assertEqual(summary["normal"]["success"],100)
            self.assertEqual(summary["normal"]["frame_accuracy"],100)
            self.assertEqual(summarize(lab.rows,"Victory"),{})

    def test_missing_hand_counts_wrong_and_missing_latency_is_blank(self):
        metrics=TrialMetrics("Victory")
        gate=StableGestureGate()
        for i in range(30):
            metrics.observe(Observation(),gate.update(None,0,i/30),i/30)
        row=metrics.row("low_light",1)
        self.assertEqual(row["frames"],30)
        self.assertEqual(row["detection_percent"],0)
        self.assertEqual(row["frame_accuracy_percent"],0)
        self.assertEqual(row["success"],0)
        self.assertEqual(row["first_correct_seconds"],"")

    def test_wrong_command_makes_trial_unsuccessful(self):
        metrics=TrialMetrics("Victory")
        gate=StableGestureGate()
        for name in ("Open_Palm","Victory"):
            for _ in range(5):
                observation=Observation(name,0.9,[object()])
                metrics.observe(observation,gate.update(name,0.9),0.2)
        row=metrics.row("normal",1)
        self.assertEqual(row["correct_events"],1)
        self.assertEqual(row["wrong_events"],1)
        self.assertEqual(row["success"],0)

    def test_cancel_and_expired_without_frames_do_not_save(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"trials.csv"
            lab=ExperimentLab(path)
            lab.start(0)
            lab.cancel()
            lab.tick(10)
            self.assertFalse(path.exists())
            lab.start(10)
            lab.tick(17)
            self.assertFalse(path.exists())


if __name__=="__main__":
    unittest.main()
