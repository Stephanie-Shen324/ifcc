#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import gzip
import os
import pickle
import time
import torch
from tqdm import tqdm
from clinicgen.data.image2text import _CaptioningData, _RadiologyReportData

import json

"""
A brief view of a document in the IU XRAY customized dataset.


 {"id": "CXR2384_IM-0942", 
 "report": "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.",
  "image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"], 
  "split": "train"}

"""


class IUXRAYData(_RadiologyReportData):
    IMAGE_NUM = 6091
    LABEL_CHEXPERT = 'chexpert'
    CHEXPERT_MAP = [13, 4, 1, 7, 6, 3, 2, 9, 0, 10, 8, 11, 5, 12]

    CHEXPERT_PATH = 'mimic-cxr-2.0.0-chexpert.csv.gz'
    META_PATH = 'mimic-cxr-2.0.0-metadata.csv.gz'
    SECTIONED_PATH = 'mimic_cxr_sectioned.csv.gz'
    SPLITS_PATH = 'mimic-cxr-2.0.0-split.csv.gz'

    def __init__(self, root, section='findings', split=None, target_transform=None, cache_image=False, cache_text=True,
                 multi_image=1, img_mode='center', img_augment=False, single_image_doc=False, dump_dir=None,
                 filter_reports=True):
        if not cache_text:
            raise ValueError('IU-XRAY data only supports cached texts')
        super().__init__(root, section, split, cache_image, cache_text, multi_image=multi_image,
                         single_image_doc=single_image_doc, dump_dir=dump_dir)
        pre_transform, self.transform = IUXRAYData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = target_transform
        self.chexpert_labels_path = os.path.join(root, 'mimic-cxr-jpg', '2.0.0', self.CHEXPERT_PATH)

        annotation = json.loads(open('/content/iu_xray_resized/annotation.json', 'r').read())
        texts_train = annotation['train']
        texts_val = annotation['val']
        texts_test = annotation['test']
        self.texts = texts_train + texts_val + texts_test

        self.view_positions = {}

        # read files as f
        for report in self.texts:
            dicom_id = report['id']
            self.view_positions[dicom_id] = 256  # HARD CODE 256

        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes(filter_reports)
                return
        # assume done
        #########################

        sections = {}
        for row in self.texts:
            study_id = str(row['id']) #edited
            report = {'report': row['report']}
            sections[study_id] = gzip.compress(pickle.dumps(report))  # ? assume row[0] is the study_id
        # assume done
        #########################

        interval = 1000
        with tqdm(total=self.IMAGE_NUM) as pbar:
            pbar.set_description('Data ({0})'.format(split))
            count = 0
            for row in self.texts:
                if split is None or split == row['split']:  # split
                    did = row['id']  # dicom_id  #not used
                    sid = row['id']  # study_id #edited
                    pid = row['id']  # subject_id -> patient_id  #not used 
                    self.ids.append(did) #not used
                    self.doc_ids.append(sid)

                    image_path = row['image_path'][0]

                    # image
                    #Todo: image_path in iu xray is list of 2 items
                    image = os.path.join(root, 'images', image_path)  # todo: modify the path
                    if cache_image:
                        image = self.bytes_image(image, pre_transform)
                    # report
                    # assume cache_text = True
                    # report = os.path.join(root, 'mimic-cxr', '2.0.0', 'files', 'p{0}'.format(pid[:2]), 'p' + pid,
                    #                       's{0}.txt'.format(sid))

                    if cache_text:
                        sid =  str(sid) #edited
                        report = sections[sid] if sid in sections else gzip.compress(pickle.dumps({}))
                        if sid not in sections:
                            print('{} not in sections'.format(sid))
                    self.samples.append((image, report))
                    # image: image_path, report: report_path
                    self.targets.append(report)
                count += 1
                if count >= interval:
                    pbar.update(count)
                    count = 0
            if count > 0:
                pbar.update(count)
        # assume done
        #########################

        if dump_dir is not None:
            self.dump()
        self.pre_processes(filter_reports)

    def __getitem__(self, index):
        rid, sample, target, _ = super().__getitem__(index)
        # View position features
        if self.multi_image > 1:
            vp = [self.view_position_embedding(self.view_positions[iid]) for iid in self.image_ids[index]]
            vp = [p.unsqueeze(dim=0) for p in vp]
            if len(vp) > self.multi_image:
                vp = vp[:self.multi_image]
            elif len(vp) < self.multi_image:
                first_vp = vp[0]
                for _ in range(self.multi_image - len(vp)):
                    vp.append(first_vp.new_zeros(first_vp.size()))
            vp = torch.cat(vp, dim=0)
        else:
            vp = self.view_position_embedding(self.view_positions[rid])
        return rid, sample, target, vp

    @classmethod
    def get_transform(cls, cache_image=False, mode='center', augment=False):
        return cls._transform(cache_image, 224, mode, augment) #what is 224?

    def compare_texts(self, text1, text2):
        if 'study' in text1 and 'study' in text2:
            return text1['study'] == text2['study']
        else:
            return True

    def decompress_text(self, text):
        return pickle.loads(gzip.decompress(text))

    # def extract_section(self, text):
    #     if self.section in text:
    #         return text[self.section].replace('\n', ' ')
    #     else:
    #         return ''
    def extract_section(self, text):
        if self.section in text:
            return text[self.section].replace('\n', ' ')
        else:
            try:
                return text['report'].replace('\n', ' ')
            except:
                return ''

    def pre_processes(self, filter_reports):
        if filter_reports:
            self.filter_empty_reports()
        if self.multi_image > 1:
            self.convert_to_multi_images()
        elif self.single_image_doc:
            self.convert_to_single_image()
        self.pre_transform_texts(self.split)

