import argparse
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from src.dataset_retrieval import Sketchy


def build_opts(args):
    return SimpleNamespace(
        data_dir=args.data_dir,
        max_size=args.max_size,
        data_split=args.data_split,
        triplet_mode=args.triplet_mode,
        match_instance_by_stem=args.match_instance_by_stem,
    )


def print_split_header(name, dataset):
    print(f'[{name}] categories: {len(dataset.all_categories)}')
    preview = ', '.join(sorted(dataset.all_categories)[:10])
    print(f'[{name}] first categories: {preview}')
    print(f'[{name}] sketches: {len(dataset.all_sketches_path)}')
    print(f'[{name}] photos: {sum(len(v) for v in dataset.all_photos_path.values())}')


def analyze_matching(name, dataset, examples):
    sketches_per_category = defaultdict(int)
    matched_per_category = defaultdict(int)
    missing_per_category = defaultdict(int)
    insufficient_negative_categories = []
    duplicate_photo_stems = defaultdict(int)
    missing_examples = []

    for category in dataset.all_categories:
        photo_stem_map = dataset.photo_stem_to_paths[category]
        duplicate_photo_stems[category] = sum(1 for paths in photo_stem_map.values() if len(paths) > 1)
        if dataset.opts.triplet_mode == 'hard' and len(dataset.all_photos_path[category]) < 2:
            insufficient_negative_categories.append(category)

    for sketch_path in dataset.all_sketches_path:
        category = os.path.basename(os.path.dirname(sketch_path))
        filename = os.path.basename(sketch_path)
        query_instance_id = dataset.instance_id_from_filename(filename)
        matched_paths = dataset.photo_stem_to_paths[category].get(query_instance_id, [])

        sketches_per_category[category] += 1
        if matched_paths:
            matched_per_category[category] += 1
        else:
            missing_per_category[category] += 1
            if len(missing_examples) < examples:
                missing_examples.append((category, filename, query_instance_id))

    total_sketches = len(dataset.all_sketches_path)
    total_matched = sum(matched_per_category.values())
    total_missing = total_sketches - total_matched
    match_rate = 0.0 if total_sketches == 0 else 100.0 * total_matched / total_sketches

    print(f'[{name}] matched sketches: {total_matched}/{total_sketches} ({match_rate:.2f}%)')
    print(f'[{name}] unmatched sketches: {total_missing}')

    categories_with_missing = sorted(
        [category for category, count in missing_per_category.items() if count > 0],
        key=lambda category: (-missing_per_category[category], category),
    )
    if categories_with_missing:
        print(f'[{name}] categories with missing positive matches: {len(categories_with_missing)}')
        for category in categories_with_missing[:10]:
            print(
                f'  - {category}: missing {missing_per_category[category]} / {sketches_per_category[category]} sketches'
            )
    else:
        print(f'[{name}] all sketches found a positive photo by stem matching')

    if insufficient_negative_categories:
        preview = ', '.join(sorted(insufficient_negative_categories)[:10])
        print(f'[{name}] categories with <2 photos (hard negative impossible): {preview}')
    else:
        print(f'[{name}] every category has at least 2 photos for hard negatives')

    categories_with_duplicate_stems = sorted(
        [category for category, count in duplicate_photo_stems.items() if count > 0],
        key=lambda category: (-duplicate_photo_stems[category], category),
    )
    if categories_with_duplicate_stems:
        print(f'[{name}] categories with duplicate photo stems: {len(categories_with_duplicate_stems)}')
        for category in categories_with_duplicate_stems[:10]:
            print(f'  - {category}: {duplicate_photo_stems[category]} duplicated stems')
    else:
        print(f'[{name}] no duplicate photo stems detected')

    if missing_examples:
        print(f'[{name}] sample unmatched sketches:')
        for category, filename, query_instance_id in missing_examples:
            print(f'  - {category}: {filename} -> expected photo stem {query_instance_id}')


def print_triplet_examples(name, dataset, examples):
    if len(dataset.all_sketches_path) == 0:
        print(f'[{name}] dataset is empty')
        return

    print(f'[{name}] sample triplets:')
    np.random.seed(0)
    sample_indices = np.linspace(0, len(dataset.all_sketches_path) - 1, num=min(examples, len(dataset.all_sketches_path)), dtype=int)
    for index in sample_indices:
        sketch_path = dataset.all_sketches_path[index]
        category = os.path.basename(os.path.dirname(sketch_path))
        filename = os.path.basename(sketch_path)
        query_instance_id = dataset.instance_id_from_filename(filename)
        positive_path = dataset.sample_positive_photo(category, filename)
        negative_path = dataset.sample_negative_photo(category, positive_path)

        positive_id = dataset.photo_id_from_path(positive_path)
        negative_id = dataset.photo_id_from_path(negative_path)
        negative_category = os.path.basename(os.path.dirname(negative_path))

        print(
            f'  - {category} | sketch={filename} | query_id={query_instance_id} | '
            f'positive={os.path.basename(positive_path)} | positive_match={positive_id == query_instance_id} | '
            f'negative={os.path.basename(negative_path)} | negative_same_category={negative_category == category} | '
            f'negative_diff_instance={negative_id != positive_id}'
        )


def main():
    parser = argparse.ArgumentParser(description='Inspect FG-SBIR dataset setup')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_size', type=int, default=224)
    parser.add_argument('--data_split', type=float, default=-1.0)
    parser.add_argument('--triplet_mode', type=str, default='hard', choices=['category', 'hard'])
    parser.add_argument('--match_instance_by_stem', dest='match_instance_by_stem', action='store_true')
    parser.add_argument('--no_match_instance_by_stem', dest='match_instance_by_stem', action='store_false')
    parser.add_argument('--examples', type=int, default=8)
    parser.set_defaults(match_instance_by_stem=True)
    args = parser.parse_args()

    opts = build_opts(args)
    transform = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, transform, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, transform, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    print('=== DATASET DEBUG ===')
    print(f'data_dir: {args.data_dir}')
    print(f'triplet_mode: {args.triplet_mode}')
    print(f'match_instance_by_stem: {args.match_instance_by_stem}')
    print()

    for name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        print_split_header(name, dataset)
        analyze_matching(name, dataset, args.examples)
        print_triplet_examples(name, dataset, args.examples)
        print()


if __name__ == '__main__':
    main()
