import unittest

from rag_utils import extract_answer, format_source_label


class RagUtilsTests(unittest.TestCase):
    def test_extract_answer_removes_echoed_prompt(self):
        prompt = "Context: policy text\nQuestion: leave"
        self.assertEqual(extract_answer(prompt + "\nAnswer: 20 days", prompt), "Answer: 20 days")

    def test_extract_answer_preserves_generation_without_echo(self):
        self.assertEqual(extract_answer("  20 days  ", "prompt"), "20 days")

    def test_source_label_keeps_filename_and_page_only(self):
        label = format_source_label(
            {"source": "/private/machine/policies/leave.pdf", "page": 4, "employee": "private"}
        )
        self.assertEqual(label, "leave.pdf (page 4)")
        self.assertNotIn("/private/machine", label)
        self.assertNotIn("employee", label)

    def test_source_label_handles_windows_paths_and_missing_metadata(self):
        self.assertEqual(format_source_label({"source": r"C:\\policies\\travel.pdf"}), "travel.pdf")
        self.assertEqual(format_source_label(None), "Unknown source")


if __name__ == "__main__":
    unittest.main()
