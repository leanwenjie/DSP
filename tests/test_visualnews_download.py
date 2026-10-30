"""Range access must preserve byte positions and reject ignored Range headers."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_visualnews as downloader


class RangeTest(unittest.TestCase):
    def test_ranges_cache_and_rejection(self):
        data = bytes(range(32))
        class Session:
            status = 206
            calls = 0
            def get(self, url, headers, timeout):
                self.calls += 1
                start,end = map(int,headers['Range'].split('=')[1].split('-'))
                class Response:
                    status_code = self.status
                    content = data[start:end+1]
                    headers = {'Content-Range':f'bytes {start}-{end}/32'}
                    def raise_for_status(self): pass
                return Response()
        with patch.object(downloader,'TOTAL',32), patch.object(downloader,'BLOCK',8):
            session = Session(); archive = downloader.RemoteArchive(session)
            self.assertEqual(archive.read(0,4),data[:4])
            self.assertEqual(archive.read(4,9),data[4:13])
            self.assertEqual(session.calls,2)
            session.status = 200
            with self.assertRaises(ValueError):archive.read(24,4)
