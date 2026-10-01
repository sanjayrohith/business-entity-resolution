import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
from core import *

class Tests(unittest.TestCase):
    def test_singletons(self):
        self.assertEqual(entity_metric(set(),set())[2],1)
        self.assertEqual(entity_metric(set(),{'S2-a'})[2],0)
        self.assertEqual(entity_metric({'S2-a'},set())[2],0)
    def test_official_example(self):
        self.assertAlmostEqual(entity_metric({'S2-a','S3-b'},{'S2-a','S3-b','S3-c'})[2],5/7)
    def test_macro_not_micro(self):
        t={'S1-a':set(),'S1-b':{'S2-a','S3-b'}}
        self.assertAlmostEqual(metrics(t,{'S1-a':set(),'S1-b':set()})['macro_f0_5'],.5)
    def test_order_duplicates_validity(self):
        self.assertEqual(parse_ids('S2-a,S3-b'),parse_ids('S3-b,S2-a'))
        for x in ['S2-a,S2-a','S1-a','S2-a,']:
            with self.assertRaises(ValueError): parse_ids(x)
        with self.assertRaises(ValueError): parse_ids('S2-a',{'S3-a'})
    def test_unicode_deterministic(self):
        self.assertEqual(norm(' École—SARL! '),'école sarl')
        self.assertEqual(fold(norm('École')),'ecole')
        self.assertEqual(norm('कृष्णा'), 'कृष्णा')
        self.assertEqual(fold('कृष्णा'),'कृष्णा')
        self.assertEqual(norm(norm('A—B  LLC')),norm('A—B  LLC'))
        self.assertEqual(suffix('co operative ltd'),'co operative')
    def test_tsv_multiple_empty_and_commas(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.tsv'; data={'S1-a':{'S2-1','S3-2'},'S1-b':set()}
            write_sets(p,data,'matched_entity_ids')
            self.assertEqual(read_sets(p,'matched_entity_ids',data,{'S2-1','S3-2'}),data)
            p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\t"A"\t1, Road\tFrance\n',encoding='utf-8')
            self.assertEqual(next(rows(p))['business_address'],'1, Road')
    def test_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.tsv'; p.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-a\t\n')
            with self.assertRaises(ValueError): read_sets(p,'matched_entity_ids')
    def test_empty_or_malformed_tsv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.tsv'
            for content in ['', 'source1_entity_id,matched_entity_ids\n', 'source1_entity_id\tmatched_entity_ids\nS1-a\n']:
                p.write_text(content)
                with self.assertRaises(ValueError): read_sets(p,'matched_entity_ids')
    def test_no_top_one(self):
        self.assertEqual(entity_metric({'S2-a','S2-b','S3-c'},{'S2-a','S2-b','S3-c'})[2],1)
    def test_official_validator_multiple_and_empty(self):
        root=Path(__file__).resolve().parents[3]
        if not (root/'utils/validate_submission.py').exists():self.skipTest('Organizer validator (utils/validate_submission.py) is not redistributed')
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for src,ids in [(1,['S1-a','S1-b']),(2,['S2-a','S2-b']),(3,['S3-a'])]:
                (d/f'test_source{src}.tsv').write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'+''.join(f'{e}\tSynthetic\t\tUnseen\n' for e in ids),encoding='utf-8')
            data={'S1-a':{'S2-a','S2-b','S3-a'},'S1-b':set()}
            write_sets(d/'local_predictions.tsv',data,'matched_entity_ids');write_sets(d/'local_candidates.tsv',data,'candidate_entity_ids')
            p=subprocess.run([sys.executable,str(root/'utils/validate_submission.py'),'--matching',str(d/'local_predictions.tsv'),'--candidate',str(d/'local_candidates.tsv'),'--test-dir',str(d),'--check-ids'],capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertNotIn('WARNING:',p.stdout)

if __name__=='__main__': unittest.main()
