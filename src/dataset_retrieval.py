import os
import glob
import math
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

        self.all_categories = sorted(os.listdir(os.path.join(self.opts.data_dir, 'sketch')))
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
            self.all_sketches_path.extend(sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png'))))
            photo_paths = sorted(glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg')))
            self.all_photos_path[category] = photo_paths
            self.photo_stem_to_paths[category] = {}
            for photo_path in photo_paths:
                photo_stem = os.path.splitext(os.path.basename(photo_path))[0]
                self.photo_stem_to_paths[category].setdefault(photo_stem, []).append(photo_path)
        self.apply_debug_filters()
        self.build_index_maps()

    def apply_debug_filters(self):
        debug_category = getattr(self.opts, 'debug_category', '').strip()
        debug_num_categories = max(0, int(getattr(self.opts, 'debug_num_categories', 0)))
        debug_num_photos_per_category = max(0, int(getattr(self.opts, 'debug_num_photos_per_category', 0)))
        debug_num_sketches_per_category = max(0, int(getattr(self.opts, 'debug_num_sketches_per_category', 0)))

        if debug_category:
            self.all_categories = [category for category in self.all_categories if category == debug_category]
        elif debug_num_categories > 0:
            self.all_categories = self.all_categories[:debug_num_categories]

        if not self.all_categories:
            self.all_sketches_path = []
            self.all_photos_path = {}
            self.photo_stem_to_paths = {}
            return

        filtered_photos_path = {}
        filtered_photo_stem_to_paths = {}
        for category in self.all_categories:
            photo_paths = list(self.all_photos_path[category])
            if debug_num_photos_per_category > 0:
                photo_paths = photo_paths[:debug_num_photos_per_category]
            filtered_photos_path[category] = photo_paths
            filtered_photo_stem_to_paths[category] = {}
            for photo_path in photo_paths:
                photo_stem = os.path.splitext(os.path.basename(photo_path))[0]
                filtered_photo_stem_to_paths[category].setdefault(photo_stem, []).append(photo_path)

        sketches_by_category = {category: [] for category in self.all_categories}
        for sketch_path in self.all_sketches_path:
            category = os.path.basename(os.path.dirname(sketch_path))
            if category in sketches_by_category:
                sketches_by_category[category].append(sketch_path)
        filtered_sketches_path = []
        for category in self.all_categories:
            sketch_paths = []
            for sketch_path in sketches_by_category[category]:
                filename = os.path.basename(sketch_path)
                if not self.opts.match_instance_by_stem:
                    sketch_paths.append(sketch_path)
                    continue
                has_positive = any(
                    sketch_stem in filtered_photo_stem_to_paths[category]
                    for sketch_stem in self.candidate_instance_stems(filename)
                )
                if has_positive:
                    sketch_paths.append(sketch_path)
            if debug_num_sketches_per_category > 0:
                sketch_paths = sketch_paths[:debug_num_sketches_per_category]
            filtered_sketches_path.extend(sketch_paths)

        self.all_sketches_path = filtered_sketches_path
        self.all_photos_path = filtered_photos_path
        self.photo_stem_to_paths = filtered_photo_stem_to_paths

    def build_index_maps(self):
        self.category_to_indices = {category: [] for category in self.all_categories}
        self.category_to_instance_to_indices = {category: {} for category in self.all_categories}
        for index, sketch_path in enumerate(self.all_sketches_path):
            category = os.path.basename(os.path.dirname(sketch_path))
            filename = os.path.basename(sketch_path)
            instance_id = self.instance_id_from_filename(filename)
            self.category_to_indices.setdefault(category, []).append(index)
            self.category_to_instance_to_indices.setdefault(category, {}).setdefault(instance_id, []).append(index)

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


class CategoryInstanceBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, categories_per_batch, instances_per_category):
        self.dataset = dataset
        self.categories_per_batch = categories_per_batch
        self.instances_per_category = instances_per_category
        self.batch_size = categories_per_batch * instances_per_category

        self.eligible_categories = [
            category for category in dataset.all_categories
            if dataset.category_to_instance_to_indices.get(category)
        ]
        if not self.eligible_categories:
            raise ValueError('no eligible categories found for CategoryInstanceBatchSampler')

        self.num_batches = max(1, math.ceil(len(self.dataset) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng()
        num_categories = len(self.eligible_categories)
        category_replace = num_categories < self.categories_per_batch

        for _ in range(self.num_batches):
            selected_categories = rng.choice(
                self.eligible_categories,
                size=self.categories_per_batch,
                replace=category_replace,
            ).tolist()

            batch_indices = []
            for category in selected_categories:
                instance_to_indices = self.dataset.category_to_instance_to_indices[category]
                instance_ids = sorted(instance_to_indices.keys())
                instance_replace = len(instance_ids) < self.instances_per_category
                selected_instances = rng.choice(
                    instance_ids,
                    size=self.instances_per_category,
                    replace=instance_replace,
                ).tolist()
                for instance_id in selected_instances:
                    candidate_indices = instance_to_indices[instance_id]
                    batch_indices.append(int(rng.choice(candidate_indices)))

            rng.shuffle(batch_indices)
            yield batch_indices


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
