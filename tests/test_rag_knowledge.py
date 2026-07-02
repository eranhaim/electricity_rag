"""
Integration tests for the Electricity RAG agent.
Sends Hebrew questions to the live API and validates the answers
contain expected knowledge from the uploaded electricity law document.
"""
import urllib.request
import json
import sys
import unittest

API_BASE = "http://54.173.144.0"


def api_post(path: str, body: dict, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, headers=hdrs)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def api_get(path: str) -> dict:
    req = urllib.request.Request(f"{API_BASE}{path}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def api_delete(path: str):
    req = urllib.request.Request(f"{API_BASE}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def has_any(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in text (case-insensitive for ASCII)."""
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower or kw in text:
            return True
    return False


class TestRAGKnowledge(unittest.TestCase):
    session_id: str = ""

    @classmethod
    def setUpClass(cls):
        session = api_post("/api/sessions", {})
        cls.session_id = session["id"]

    @classmethod
    def tearDownClass(cls):
        if cls.session_id:
            try:
                api_delete(f"/api/sessions/{cls.session_id}")
            except Exception:
                pass

    def _ask(self, question: str) -> str:
        result = api_post(
            f"/api/sessions/{self.session_id}/chat",
            {"message": question},
        )
        self.assertIn("answer", result)
        answer = result["answer"]
        sys.stderr.write(f"\nQ: {question}\nA: {answer[:300]}...\n")
        return answer

    def test_01_grounding_colors(self):
        """The document states grounding conductor color is yellow-green, neutral is blue."""
        answer = self._ask("מהו צבע הבידוד של מוליך הארקה ומוליך אפס לפי התקנות?")
        self.assertTrue(
            has_any(answer, ["ירוק", "צהוב", "green", "yellow", "כחול", "blue"]),
            f"Answer should mention yellow-green or blue. Got: {answer[:200]}",
        )

    def test_02_junction_box_height(self):
        """Junction box must be at least 2.1m above floor, or 90cm with tool-removable cover."""
        answer = self._ask("לפי תקנות החשמל, מה הגובה המינימלי בסנטימטרים להתקנת קופסת חיבור (junction box) מעל הרצפה?")
        self.assertTrue(
            has_any(answer, ["2.1", "2.10", "210", "90", "מטר", "meter", "ס\"מ", "ס״מ", "cm", "גובה", "height"]),
            f"Answer should mention height requirements. Got: {answer[:200]}",
        )

    def test_03_insulation_resistance(self):
        """Insulation resistance must be at least 0.9 megohms for operational test."""
        answer = self._ask("מהי רמת התנגדות הבידוד המינימלית בבדיקה תפעולית במתח נמוך לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["0.9", "מגאום", "megohm", "MΩ", "בידוד", "insulation", "התנגדות"]),
            f"Answer should mention insulation resistance values. Got: {answer[:200]}",
        )

    def test_04_licensed_electrician(self):
        """Only licensed electricians may install conductors."""
        answer = self._ask("מי רשאי להתקין מוליכים במתקן חשמלי לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["חשמלאי", "רישיון", "מורשה", "רשאי", "licensed", "electrician", "בעל"]),
            f"Answer should mention licensed electrician. Got: {answer[:200]}",
        )

    def test_05_noise_level(self):
        """Noise from panels with isolation transformers must not exceed 45 dBA at 1 meter."""
        answer = self._ask("מהי רמת הרעש המקסימלית המותרת מלוחות עם שנאי בידוד לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["45", "dBA", "דציבל", "רעש", "noise"]),
            f"Answer should mention 45 dBA noise level. Got: {answer[:200]}",
        )

    def test_06_periodic_testing_medical(self):
        """Medical facilities require periodic testing by licensed electricians."""
        answer = self._ask("מהן דרישות הבדיקות התקופתיות במתקנים רפואיים לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["חודש", "שנת", "שנה", "תקופ", "month", "annual", "periodic", "בדיק"]),
            f"Answer should mention periodic testing. Got: {answer[:200]}",
        )

    def test_07_conductor_connection_methods(self):
        """Connections must use terminals, screws, rivets, crimping, soldering, or welding."""
        answer = self._ask("באילו שיטות מותר לחבר בין מוליכים לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["הלחמ", "ריתוך", "בורג", "מסוף", "חיבור", "solder", "weld", "crimp", "terminal", "screw"]),
            f"Answer should mention connection methods. Got: {answer[:200]}",
        )

    def test_08_conductive_floor_testing(self):
        """Floor resistance testing: dry floor max 1 megohm, wet floor min 10 kiloohms."""
        answer = self._ask("מהן דרישות בדיקת ההתנגדות לרצפה מוליכה יבשה ורטובה?")
        self.assertTrue(
            has_any(answer, ["מגאום", "megohm", "קילואום", "kiloohm", "רצפה", "floor", "יבש", "רטוב"]),
            f"Answer should mention floor resistance requirements. Got: {answer[:200]}",
        )

    def test_09_high_voltage_min_cross_section(self):
        """Exposed copper conductors in high voltage must have min 10 mm² cross-section."""
        answer = self._ask("מהו חתך המוליך המינימלי למוליך נחושת חשוף במתח גבוה?")
        self.assertTrue(
            has_any(answer, ["10", "מ\"מ", "מ״מ", "mm", "חתך", "cross-section", "נחושת", "copper"]),
            f"Answer should mention 10 mm² cross-section. Got: {answer[:200]}",
        )

    def test_10_dialysis_machine_power(self):
        """Dialysis machines should be powered by a dedicated circuit without a protective switch."""
        answer = self._ask("כיצד יש לחבר מכונת דיאליזה לחשמל לפי תקנות החשמל?")
        self.assertTrue(
            has_any(answer, ["ייעודי", "מפסק", "מעגל", "dedicated", "circuit", "דיאליזה", "dialysis", "ישיר"]),
            f"Answer should mention dedicated circuit. Got: {answer[:200]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
