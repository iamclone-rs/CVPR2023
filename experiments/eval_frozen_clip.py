import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.clip import clip
from src.dataset_retrieval import Sketchy


def build_opts(args):
    return SimpleNamespace(
        data_dir=args.data_dir,
        max_size=args.max_size,
        data_split=args.data_split,
        triplet_mode=args.triplet_mode,
        match_instance_by_stem=args.match_instance_by_stem,
        debug_category=args.debug_category,
        debug_num_categories=args.debug_num_categories,
        debug_num_photos_per_category=args.debug_num_photos_per_category,
        debug_num_sketches_per_category=args.debug_num_sketches_per_category,
    )


def evaluate_fine_grained(model, dataloader, device, query_source='sketch'):
    query_features = []
    gallery_features = []
    categories = []
    query_instance_ids = []
    photo_ids = []
    img_paths = []

    with torch.no_grad():
        for batch in dataloader:
            sk_tensor, img_tensor = batch[0].to(device), batch[1].to(device)
            category = list(batch[3])
            query_instance_id = list(batch[5])
            photo_id = list(batch[6])
            img_path = list(batch[7])

            sk_feat = F.normalize(model.encode_image(sk_tensor), dim=-1)
            img_feat = F.normalize(model.encode_image(img_tensor), dim=-1)

            if query_source == 'sketch':
                query_features.append(sk_feat.cpu())
            elif query_source == 'photo':
                query_features.append(img_feat.cpu())
            else:
                raise ValueError('unsupported query_source: {}'.format(query_source))
            gallery_features.append(img_feat.cpu())
            categories.extend(category)
            query_instance_ids.extend(query_instance_id)
            photo_ids.extend(photo_id)
            img_paths.extend(img_path)

    if not query_features:
        return 0.0, 0.0

    query_feat_all = torch.cat(query_features, dim=0)
    gallery_feat_all = torch.cat(gallery_features, dim=0)
    categories = np.array(categories)

    unique_gallery_indices = []
    seen_gallery_keys = set()
    for idx, (category, photo_id, img_path) in enumerate(zip(categories, photo_ids, img_paths)):
        gallery_key = (category, photo_id, img_path)
        if gallery_key in seen_gallery_keys:
            continue
        seen_gallery_keys.add(gallery_key)
        unique_gallery_indices.append(idx)

    gallery_feat = gallery_feat_all[unique_gallery_indices]
    gallery_category = categories[unique_gallery_indices]
    gallery_photo_id = np.array(photo_ids, dtype=object)[unique_gallery_indices]

    category_to_gallery_indices = {}
    for idx, category in enumerate(gallery_category):
        category_to_gallery_indices.setdefault(category, []).append(idx)

    acc_at_1 = torch.zeros(len(query_feat_all), dtype=torch.float32)
    acc_at_5 = torch.zeros(len(query_feat_all), dtype=torch.float32)
    for idx, query_feat in enumerate(query_feat_all):
        category = categories[idx]
        true_photo_id = query_instance_ids[idx]
        category_gallery_indices = category_to_gallery_indices.get(category, [])
        if not category_gallery_indices:
            continue

        category_gallery_feat = gallery_feat[category_gallery_indices]
        similarities = torch.matmul(category_gallery_feat, query_feat)
        ranking = torch.argsort(similarities, descending=True)
        ranked_photo_ids = gallery_photo_id[np.array(category_gallery_indices)[ranking.numpy()]]
        acc_at_1[idx] = float(true_photo_id in ranked_photo_ids[:1])
        acc_at_5[idx] = float(true_photo_id in ranked_photo_ids[:5])

    return acc_at_1.mean().item(), acc_at_5.mean().item()


def main():
    parser = argparse.ArgumentParser(description='Evaluate frozen CLIP on FG-SBIR retrieval metric')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--query_source', type=str, default='sketch', choices=['sketch', 'photo'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_size', type=int, default=224)
    parser.add_argument('--data_split', type=float, default=-1.0)
    parser.add_argument('--triplet_mode', type=str, default='hard', choices=['category', 'hard'])
    parser.add_argument('--match_instance_by_stem', dest='match_instance_by_stem', action='store_true')
    parser.add_argument('--no_match_instance_by_stem', dest='match_instance_by_stem', action='store_false')
    parser.add_argument('--debug_category', type=str, default='')
    parser.add_argument('--debug_num_categories', type=int, default=0)
    parser.add_argument('--debug_num_photos_per_category', type=int, default=0)
    parser.add_argument('--debug_num_sketches_per_category', type=int, default=0)
    parser.set_defaults(match_instance_by_stem=True)
    args = parser.parse_args()

    opts = build_opts(args)
    transform = Sketchy.data_transform(opts)
    train_dataset = Sketchy(opts, transform, mode='train', return_orig=False)
    if args.split == 'train':
        dataset = train_dataset
    else:
        dataset = Sketchy(opts, transform, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load('ViT-B/32', device=device)
    model.eval()

    acc1, acc5 = evaluate_fine_grained(model, dataloader, device, query_source=args.query_source)
    print('split: {}'.format(args.split))
    print('query_source: {}'.format(args.query_source))
    print('samples: {}'.format(len(dataset)))
    print('categories: {}'.format(len(dataset.all_categories)))
    print('Frozen CLIP Acc@1: {:.4f}, Acc@5: {:.4f}'.format(acc1, acc5))


if __name__ == '__main__':
    main()
