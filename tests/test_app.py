import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from app_inference import validate_input
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def png():
    buf=io.BytesIO(); Image.new('RGB',(32,24),'white').save(buf,format='PNG');return buf.getvalue()

class InputTests(unittest.TestCase):
    def test_valid_image_and_trimmed_caption(self):
        im,caption=validate_input(png(),'  A white image.  ')
        self.assertEqual(im.size,(32,24));self.assertEqual(caption,'A white image.')
    def test_missing_and_corrupt_inputs(self):
        for content,caption in [(b'','a caption'),(png(),' '),(b'broken','a caption')]:
            with self.assertRaises(ValueError):validate_input(content,caption)
    def test_long_caption_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'token'):
            validate_input(png(),'elephant '*100)
    def test_upload_limits(self):
        with self.assertRaisesRegex(ValueError,'10 MB'):validate_input(b'x'*(10*1024*1024+1),'A caption')
        with patch('app_inference.MAX_PIXELS',10):
            with self.assertRaisesRegex(ValueError,'megapixels'):validate_input(png(),'A caption')
    def test_animation_rejected(self):
        buf=io.BytesIO();Image.new('RGB',(4,4),'red').save(buf,format='PNG',save_all=True,append_images=[Image.new('RGB',(4,4),'blue')],duration=100,loop=0)
        with self.assertRaisesRegex(ValueError,'animation'):validate_input(buf.getvalue(),'A caption')

class InterfaceTests(unittest.TestCase):
    def test_startup_and_missing_upload(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        self.assertEqual(at.selectbox[0].value,'Logistic regression')
        at.button[0].click().run()
        self.assertEqual(at.error[0].value,'Upload an image first.')
    def test_valid_submission_and_failure_messages(self):
        for flagged in (False,True):
            result=dict(flagged=flagged,score=.2,seconds=.1,model='Logistic regression',score_type='decision score',flag_when='above',threshold=.1)
            with patch('streamlit.file_uploader',return_value=io.BytesIO(png())),patch('app_inference.Detector') as detector:
                detector.return_value.predict.return_value=result
                import streamlit as st
                st.cache_resource.clear()
                at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
                at.text_area[0].set_value('A white image.');at.button[0].click().run()
                self.assertEqual(len(at.exception),0)
                messages=[x.value for x in (at.warning if flagged else at.info)]
                self.assertTrue(any(('Flagged for review' if flagged else 'No mismatch flagged') in x for x in messages))
                detector.return_value.predict.assert_called_once()
        st.cache_resource.clear()
        with patch('streamlit.file_uploader',return_value=io.BytesIO(png())),patch('app_inference.Detector',side_effect=RuntimeError('missing')):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
            at.text_area[0].set_value('A white image.');at.button[0].click().run()
            self.assertIn('Inference is unavailable',at.error[0].value)
            self.assertEqual(len(at.exception),0)
