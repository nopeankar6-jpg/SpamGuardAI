import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from preprocess import clean_text, extract_email, prepare_input, load_clean_data, process_archives


def email(subject, body, *, html=False):
    content_type = "text/html" if html else "text/plain"
    return (f"Subject: {subject}\nContent-Type: {content_type}; charset=utf-8\n\n{body}").encode("utf-8")


def create_archive(path, records):
    with tarfile.open(path, "w:bz2") as tar:
        for idx, data in enumerate(records):
            info = tarfile.TarInfo(name=f"messages/{idx:05d}.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class PreprocessingTests(unittest.TestCase):
    def test_clean_and_extract(self):
        self.assertEqual(clean_text("  Hello \n  world  "), "Hello world")
        subject, body, text = extract_email(email("***SPAM*** FREE OFFER", "Click here"))
        self.assertEqual(subject, "FREE OFFER")
        self.assertEqual(text, "FREE OFFER Click here")
        self.assertEqual(prepare_input(subject, body), text)
        _, body2, text2 = extract_email(email("Hi", "<p>Hello</p><script>evil()</script><b>friend</b>", html=True))
        self.assertIn("Hello", body2)
        self.assertIn("friend", text2)
        self.assertNotIn("evil", text2)
        self.assertEqual(clean_text(None), "")

    def test_underscore_archive_stats_and_duplicates(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            raw = root / "raw"
            raw.mkdir()
            create_archive(raw / "20021010_easy_ham_tar.bz2", [
                email("Hello", "Project meeting"),
                email("Hello", "Project meeting"),
                email("Shared", "Conflict body"),
            ])
            create_archive(raw / "20030228_spam_tar.bz2", [
                email("***SPAM*** Prize", "Claim now"),
                email("Shared", "Conflict body"),
            ])
            df, stats = process_archives(raw, root / "out.csv", root / "stats.json")
            self.assertEqual(stats["valid_before_dedup"], 5)
            self.assertEqual(stats["contradictory_rows_removed"], 2)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df["label"]), {"spam", "ham"})
            loaded = load_clean_data(root / "out.csv")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(json.loads((root / "stats.json").read_text(encoding="utf-8"))["total_after_cleaning"], 2)


if __name__ == "__main__":
    unittest.main()
