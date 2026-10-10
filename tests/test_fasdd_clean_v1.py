"""Meaningful selection/decoder checks; no training or GPU use."""
import importlib.util
import base64
from pathlib import Path
import tempfile
import unittest
from PIL import Image

spec=importlib.util.spec_from_file_location('clean_builder',Path(__file__).resolve().parents[1]/'scripts/prepare_fasdd_clean_v1.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def row(identity,split,byte,pixel,dataset='fasdd',error=None):
    return {'source_id':identity,'split':split,'sha256':byte,'decoded_sha256':pixel,
            'dataset':dataset,'decode_error':error}


class CleanSelectionTests(unittest.TestCase):
    def test_reference_any_split_excludes_target(self):
        records=[row('target','train','a','p'),row('ref','test','a','p','dfire')]
        kept,excluded=builder.select_records(records)
        self.assertEqual(kept,[])
        self.assertEqual(excluded[0]['reason'],'cross_dataset_byte_duplicate')

    def test_reencoded_pixels_exclude_against_reference(self):
        kept,excluded=builder.select_records([row('target','val','a','p'),row('ref','train','b','p','dfire')])
        self.assertEqual(kept,[])
        self.assertEqual(excluded[0]['reason'],'cross_dataset_decoded_duplicate')

    def test_official_split_priority_and_stable_tie(self):
        records=[row('train','train','a','p'),row('val','val','b','p'),row('test-z','test','c','p'),row('test-a','test','d','p')]
        kept,excluded=builder.select_records(records)
        self.assertEqual([r['source_id'] for r in kept],['test-a'])
        self.assertEqual(len(excluded),3)
        kept_reverse,_=builder.select_records(list(reversed(records)))
        self.assertEqual(kept_reverse,kept)

    def test_hash_similarity_never_excludes_distinct_content(self):
        records=[row('a','train','a','p'),row('b','test','b','q')]
        for r in records:r.update(phash=0,dhash=0)
        self.assertEqual(len(builder.select_records(records)[0]),2)

    def test_decode_failure_excluded(self):
        kept,excluded=builder.select_records([row('bad','train','a',None,error='bad image')])
        self.assertEqual(kept,[])
        self.assertEqual(excluded[0]['reason'],'image_decode_failure')

    def test_decoder_equivalence_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=Image.new('RGB',(16,12),(33,100,190))
            image.save(root/'a.png');image.save(root/'b.bmp')
            image.resize((8,6)).save(root/'c.png')
            records=[builder.scan_one((f.name,'fasdd','train',str(f))) for f in (root/'a.png',root/'b.bmp',root/'c.png')]
            self.assertNotEqual(records[0]['sha256'],records[1]['sha256'])
            self.assertEqual(records[0]['decoded_sha256'],records[1]['decoded_sha256'])
            self.assertNotEqual(records[0]['decoded_sha256'],records[2]['decoded_sha256'])

    def test_corrupt_image_is_not_silent_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.jpg';path.write_bytes(b'not an image')
            record=builder.scan_one(('bad','fasdd','train',str(path)))
            self.assertTrue(record['decode_error'])
            self.assertTrue(record['sha256'])

    def test_perceptual_audit_reports_without_excluding(self):
        records=[row('dfire/test/reference','test','a','p','dfire'),row('fasdd/train/target','train','b','q')]
        for r in records:
            r.update(phash=123,dhash=456,gray_std=0.0,
                     gray32=base64.b64encode(bytes([100])*1024).decode())
        with tempfile.TemporaryDirectory() as directory:
            summary=builder.audit_near(records,Path(directory)/'candidates.csv')
            self.assertEqual(summary['review_candidate_pairs'],1)
            self.assertEqual(summary['low_texture_pairs'],1)
            self.assertEqual(summary['review_status'],'pending')
        self.assertEqual(len(builder.select_records(records)[0]),1)

    def test_same_split_perceptual_redundancy_not_claimed_as_leakage(self):
        records=[row('fasdd/train/a','train','a','p'),row('fasdd/train/b','train','b','q')]
        for r in records:
            r.update(phash=123,dhash=456,gray_std=0.0,
                     gray32=base64.b64encode(bytes([100])*1024).decode())
        with tempfile.TemporaryDirectory() as directory:
            summary=builder.audit_near(records,Path(directory)/'candidates.csv')
            self.assertEqual(summary['review_candidate_pairs'],0)
        self.assertEqual(len(builder.select_records(records)[0]),2)


if __name__=='__main__':unittest.main()
