import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
WHITE_RGB = (255, 255, 255)

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.photo_stem_to_paths = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            photo_paths = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))
            self.all_photos_path[category] = photo_paths
            self.photo_stem_to_paths[category] = {}
            for photo_path in photo_paths:
                photo_stem = os.path.splitext(os.path.basename(photo_path))[0]
                self.photo_stem_to_paths[category].setdefault(photo_stem, []).append(photo_path)

    def __len__(self):
        return len(self.all_sketches_path)

    @staticmethod
    def candidate_instance_stems(filename):
        sketch_stem = os.path.splitext(filename)[0]
        candidate_stems = [sketch_stem]
        if '-' in sketch_stem:
            base_stem, suffix = sketch_stem.rsplit('-', 1)
            if suffix.isdigit():
                candidate_stems.append(base_stem)
        return candidate_stems

    @classmethod
    def instance_id_from_filename(cls, filename):
        candidate_stems = cls.candidate_instance_stems(filename)
        return candidate_stems[-1]

    @staticmethod
    def photo_id_from_path(photo_path):
        return os.path.splitext(os.path.basename(photo_path))[0]

    @staticmethod
    def load_rgb_image(path):
        image = Image.open(path)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGBA')
            canvas = Image.new('RGBA', image.size, WHITE_RGB + (255,))
            image = Image.alpha_composite(canvas, image).convert('RGB')
        else:
            image = image.convert('RGB')
        return image

    def sample_positive_photo(self, category, filename):
        if self.opts.match_instance_by_stem:
            for sketch_stem in self.candidate_instance_stems(filename):
                matched_paths = self.photo_stem_to_paths[category].get(sketch_stem, [])
                if matched_paths:
                    return np.random.choice(matched_paths)

        return np.random.choice(self.all_photos_path[category])

    def sample_negative_photo(self, category, positive_path):
        if self.opts.triplet_mode == 'hard':
            negative_pool = [path for path in self.all_photos_path[category] if path != positive_path]
            if negative_pool:
                return np.random.choice(negative_pool)

        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)
        return np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        query_instance_id = self.instance_id_from_filename(filename)

        sk_path  = filepath
        img_path = self.sample_positive_photo(category, filename)
        neg_path = self.sample_negative_photo(category, img_path)
        photo_id = self.photo_id_from_path(img_path)

        sk_data  = ImageOps.pad(self.load_rgb_image(sk_path),  size=(self.opts.max_size, self.opts.max_size), color=WHITE_RGB)
        img_data = ImageOps.pad(self.load_rgb_image(img_path), size=(self.opts.max_size, self.opts.max_size), color=WHITE_RGB)
        neg_data = ImageOps.pad(self.load_rgb_image(neg_path), size=(self.opts.max_size, self.opts.max_size), color=WHITE_RGB)

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename, query_instance_id, photo_id, img_path,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename, query_instance_id, photo_id, img_path)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, category, filename, query_instance_id, photo_id, img_path,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
