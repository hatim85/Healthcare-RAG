import unittest
from app.guardrails.pii_guard import detect_pii, mask_pii

class TestPIIGuard(unittest.TestCase):
    
    def test_masks_email_and_phone(self):
        text=(
            "Contact the patient at alex@example.com"
            "or 212-555-0101"
        )
        result=mask_pii(text)
        self.assertTrue(result["pii_detected"])
        self.assertNotIn("alex@example.com",result["text"])
        self.assertNotIn("212-555-0101",result["text"])
        
        
    def test_masks_labelled_mrn(self):
        text="Patient MRN: ABC12345 needs a chart review."
        result=mask_pii(text)
        self.assertTrue(result["pii_detected"])
        self.assertNotIn("ABC12345",result["text"])
        self.assertIn("MEDICAL_RECORD_NUMBER",result["entity_types"])
        
    def test_clean_text_remains_unchanged(self):
        text="The document describes clinical documentation."
        result=mask_pii(text)
        self.assertEqual(result["text"],text)
        self.assertFalse(result["pii_detected"])
        
    def test_detection_metadata_does_not_expose_values(self):
        text="Email: alex@example.com"
        result=detect_pii(text)
        self.assertTrue(result["pii_detected"])
        self.assertNotIn("alex@example.com",str(result))
        
if __name__=="__main__":
    unittest.main()