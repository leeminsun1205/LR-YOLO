"""Offline fixtures only: no network, no real checkpoints or Torch imports."""
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from vision_demo.releases import RELEASES,choose_member,extract_checkpoint,prepare_author_release

REPO=Path(__file__).resolve().parents[2]

class ReleaseTests(unittest.TestCase):
    def test_member_selection_and_ambiguity(self):
        names=['run/epoch_100.pth','run/best_score_epoch_95.pth']
        self.assertEqual(choose_member(names),names[1])
        self.assertEqual(choose_member(names,override=names[0]),names[0])
        for names in (['a/best_x.pth','b/best_y.pth'],['a.pth','b.pth']):
            with self.assertRaisesRegex(ValueError,'AUTHOR_CHECKPOINT_MEMBERS'):choose_member(names)
        with self.assertRaisesRegex(ValueError,'Unsafe'):choose_member(['../best_x.pth'])

    def test_all_four_configs_and_download_cache(self):
        for key in RELEASES:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                temp=Path(temp);source=temp/'source.zip'
                name=RELEASES[key].get('member','best_score_epoch_90.pth')
                with zipfile.ZipFile(source,'w') as z:
                    z.writestr('author_run/'+name,b'fake tensor fixture')
                    z.writestr('author_run/epoch_1.pth',b'other fixture')
                calls=[]
                def downloader(**kwargs):
                    calls.append(kwargs['id']);shutil.copy2(source,kwargs['output'])
                result=prepare_author_release(REPO,temp/'cache',key,downloader=downloader)
                again=prepare_author_release(REPO,temp/'cache',key,downloader=downloader)
                self.assertEqual(result,again);self.assertEqual(calls,[RELEASES[key]['file_id']])
                self.assertTrue(Path(result['config']).is_file())
                self.assertEqual(Path(result['checkpoint']).read_bytes(),b'fake tensor fixture')
                self.assertEqual(result['origin'],'author_release')
                record=json.loads((Path(result['checkpoint']).parent/'release.json').read_text())
                self.assertEqual(record['checkpoint_sha256'],result['sha256'])

    def test_raw_torch_zip_and_drive_error(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);raw=temp/'raw.bin';target=temp/'target.pth'
            with zipfile.ZipFile(raw,'w') as z:z.writestr('archive/data.pkl',b'fixture')
            self.assertEqual(extract_checkpoint(raw,target),'(raw torch ZIP checkpoint)')
            self.assertEqual(raw.read_bytes(),target.read_bytes())
            raw.write_text('<html>Drive quota error</html>')
            with self.assertRaisesRegex(ValueError,'Drive error'):extract_checkpoint(raw,target)

    def test_drive_html_error_is_not_promoted_to_reusable_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            def downloader(**kwargs):Path(kwargs['output']).write_text('<html>quota</html>')
            with self.assertRaisesRegex(ValueError,'rerun will retry'):
                prepare_author_release(REPO,temp,'voc_dark_v10',downloader=downloader)
            self.assertFalse((Path(temp)/'voc_dark_v10/author_download.bin').exists())
            self.assertFalse((Path(temp)/'voc_dark_v10/author_checkpoint.pth').exists())

    def test_existing_checkpoint_without_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'voc_dark_v10';folder.mkdir()
            p=folder/'author_checkpoint.pth';p.write_bytes(b'keep')
            with self.assertRaisesRegex(FileExistsError,'without release provenance'):
                prepare_author_release(REPO,temp,'voc_dark_v10')
            self.assertEqual(p.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
