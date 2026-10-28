import json
import tempfile
import unittest
from pathlib import Path

from agent_lab_memory.failures import classify_error, has_zero_accuracy
from agent_lab_memory.llm import extract_command
from agent_lab_memory.memory import Embedder, LongTermMemory


class CoreTests(unittest.TestCase):
    def test_extract_command_reads_fenced_payload(self):
        self.assertEqual(extract_command("```PLAN\nCompare two models.\n```", "PLAN"), "Compare two models.")


    def test_extract_command_ignores_reasoning_block(self):
        text = "<think>```PLAN\nwrong\n```</think>\n```PLAN\nright\n```"
        self.assertEqual(extract_command(text, "PLAN"), "right")


    def test_memory_persists_and_retrieves(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = LongTermMemory(str(path), embedder=Embedder(use_dense=False))
            memory.add("Report confidence intervals for model comparisons.", "insight", "classification",
                       phase="running_experiments", importance=0.9)
            memory.save()

            restored = LongTermMemory(str(path), embedder=Embedder(use_dense=False))
            found = restored.retrieve("confidence intervals for classification", phase="running_experiments")

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].kind, "insight")


    def test_read_only_memory_does_not_change_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = LongTermMemory(str(path), embedder=Embedder(use_dense=False))
            memory.add("Use a verified dataset id.", "insight", "data", phase="data_preparation")
            memory.save()
            before = json.loads(path.read_text(encoding="utf-8"))

            frozen = LongTermMemory(str(path), embedder=Embedder(use_dense=False), read_only=True)
            frozen.start_run()
            frozen.retrieve("verified dataset", phase="data_preparation")
            frozen.add("New lesson", "insight", "data")
            frozen.save()

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), before)


    def test_failure_helpers_classify_common_errors(self):
        self.assertEqual(classify_error("ModuleNotFoundError: No module named sklearn"), "dependencia_ausente")
        self.assertTrue(has_zero_accuracy("accuracy: 0.0"))
        self.assertFalse(has_zero_accuracy("accuracy: 0.72"))


if __name__ == "__main__":
    unittest.main()