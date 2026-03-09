import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad_(False)

def unfreeze_layer_norms(module):
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.LayerNorm):
            if submodule.weight is not None:
                submodule.weight.requires_grad_(True)
            if submodule.bias is not None:
                submodule.bias.requires_grad_(True)

def sample_patch_permutations(batch_size, num_patches, device):
    permutations = torch.stack(
        [torch.randperm(num_patches, device=device) for _ in range(batch_size)],
        dim=0)
    return permutations

def sample_different_patch_permutations(reference_permutations):
    batch_size, num_patches = reference_permutations.shape
    if num_patches <= 1:
        return reference_permutations.clone()

    permutations = sample_patch_permutations(
        batch_size=batch_size,
        num_patches=num_patches,
        device=reference_permutations.device)
    same_rows = torch.all(permutations == reference_permutations, dim=1)
    while same_rows.any():
        permutations[same_rows] = sample_patch_permutations(
            batch_size=int(same_rows.sum().item()),
            num_patches=num_patches,
            device=reference_permutations.device)
        same_rows = torch.all(permutations == reference_permutations, dim=1)
    return permutations

def shuffle_image_patches(images, permutations, grid_size):
    batch_size, channels, height, width = images.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError('image size must be divisible by patch shuffle grid size')

    patch_height = height // grid_size
    patch_width = width // grid_size
    patches = images.reshape(batch_size, channels, grid_size, patch_height, grid_size, patch_width)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.reshape(batch_size, grid_size * grid_size, channels, patch_height, patch_width)

    gather_index = permutations[:, :, None, None, None].expand(-1, -1, channels, patch_height, patch_width)
    shuffled = patches.gather(1, gather_index)
    shuffled = shuffled.reshape(batch_size, grid_size, grid_size, channels, patch_height, patch_width)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled = shuffled.reshape(batch_size, channels, height, width)
    return shuffled

class Model(pl.LightningModule):
    def __init__(self, categories):
        super().__init__()
        self.save_hyperparameters({"categories": list(categories)})

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        freeze_module(self.clip)
        unfreeze_layer_norms(self.clip.visual)

        # FG-SBIR uses a shared visual prompt for sketch and photo branches.
        self.common_prompt = nn.Parameter(
            self.opts.prompt_init_std * torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.train_categories = sorted(categories)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.train_categories)}
        class_prompts = [f'a photo of a {category}' for category in self.train_categories]
        self.register_buffer('class_tokens', clip.tokenize(class_prompts), persistent=False)

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.triplet_margin)
        self.patch_shuffle_loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.patch_shuffle_margin)

        self.best_metric = -1e3
        self.validation_outputs = []

    def configure_optimizers(self):
        visual_ln_params = [param for param in self.clip.visual.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam([
            {'params': visual_ln_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.common_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def load_state_dict(self, state_dict, strict=True):
        if 'common_prompt' not in state_dict and 'sk_prompt' in state_dict and 'img_prompt' in state_dict:
            state_dict = dict(state_dict)
            state_dict['common_prompt'] = 0.5 * (state_dict['sk_prompt'] + state_dict['img_prompt'])
            del state_dict['sk_prompt']
            del state_dict['img_prompt']
        return super().load_state_dict(state_dict, strict=strict)

    def category_labels(self, categories):
        labels = [self.category_to_idx[category] for category in categories]
        return torch.tensor(labels, device=self.device, dtype=torch.long)

    def encode_class_text_features(self):
        with torch.no_grad():
            text_features = self.clip.encode_text(self.class_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def classification_loss(self, feat, labels):
        feat = F.normalize(feat, dim=-1)
        text_features = self.encode_class_text_features()
        logit_scale = self.clip.logit_scale.exp().detach()
        logits = logit_scale * feat @ text_features.t()
        return F.cross_entropy(logits.float(), labels)

    def patch_shuffle_loss(self, sk_tensor, img_tensor):
        if self.opts.patch_shuffle_loss_weight <= 0:
            return sk_tensor.new_zeros(())
        if self.opts.patch_shuffle_grid_size <= 1:
            return sk_tensor.new_zeros(())

        num_patches = self.opts.patch_shuffle_grid_size ** 2
        shared_permutations = sample_patch_permutations(
            batch_size=sk_tensor.shape[0],
            num_patches=num_patches,
            device=sk_tensor.device)
        different_permutations = sample_different_patch_permutations(shared_permutations)

        sk_shared = shuffle_image_patches(sk_tensor, shared_permutations, self.opts.patch_shuffle_grid_size)
        img_shared = shuffle_image_patches(img_tensor, shared_permutations, self.opts.patch_shuffle_grid_size)
        img_different = shuffle_image_patches(img_tensor, different_permutations, self.opts.patch_shuffle_grid_size)

        sk_shared_feat = self.forward(sk_shared, dtype='sketch')
        img_shared_feat = self.forward(img_shared, dtype='image')
        img_different_feat = self.forward(img_different, dtype='image')
        return self.patch_shuffle_loss_fn(sk_shared_feat, img_shared_feat, img_different_feat)

    def forward(self, data, dtype='image'):
        feat = self.clip.encode_image(
            data, self.common_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        labels = self.category_labels(category)
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        sk_cls_loss = self.classification_loss(sk_feat, labels)
        img_cls_loss = self.classification_loss(img_feat, labels)
        cls_loss = sk_cls_loss + img_cls_loss
        patch_shuffle_loss = self.patch_shuffle_loss(sk_tensor, img_tensor)
        loss = (
            triplet_loss
            + self.opts.cls_loss_weight * cls_loss
            + self.opts.patch_shuffle_loss_weight * patch_shuffle_loss
        )

        self.log('train_triplet_loss', triplet_loss)
        self.log('train_sketch_cls_loss', sk_cls_loss)
        self.log('train_image_cls_loss', img_cls_loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_patch_shuffle_loss', patch_shuffle_loss)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        query_instance_id = batch[5]
        photo_id = batch[6]
        img_path = batch[7]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', triplet_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.validation_outputs.append((
            sk_feat.detach().cpu(),
            img_feat.detach().cpu(),
            list(category),
            list(query_instance_id),
            list(photo_id),
            list(img_path),
        ))

    def on_validation_epoch_end(self):
        Len = len(self.validation_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([self.validation_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([self.validation_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([self.validation_outputs[i][2] for i in range(Len)], []))
        all_query_instance_id = sum([self.validation_outputs[i][3] for i in range(Len)], [])
        all_photo_id = sum([self.validation_outputs[i][4] for i in range(Len)], [])
        all_img_path = sum([self.validation_outputs[i][5] for i in range(Len)], [])

        unique_gallery_indices = []
        seen_gallery_keys = set()
        for idx, (category, photo_id, img_path) in enumerate(zip(all_category, all_photo_id, all_img_path)):
            gallery_key = (category, photo_id, img_path)
            if gallery_key in seen_gallery_keys:
                continue
            seen_gallery_keys.add(gallery_key)
            unique_gallery_indices.append(idx)

        gallery_feat = F.normalize(gallery_feat_all[unique_gallery_indices], dim=-1)
        gallery_category = all_category[unique_gallery_indices]
        gallery_photo_id = np.array(all_photo_id, dtype=object)[unique_gallery_indices]

        category_to_gallery_indices = {}
        for idx, category in enumerate(gallery_category):
            category_to_gallery_indices.setdefault(category, []).append(idx)

        query_feat_all = F.normalize(query_feat_all, dim=-1)
        acc_at_1 = torch.zeros(len(query_feat_all), dtype=torch.float32)
        acc_at_5 = torch.zeros(len(query_feat_all), dtype=torch.float32)

        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            true_photo_id = all_query_instance_id[idx]
            category_gallery_indices = category_to_gallery_indices.get(category, [])
            if len(category_gallery_indices) == 0:
                continue

            category_gallery_feat = gallery_feat[category_gallery_indices]
            similarities = torch.matmul(category_gallery_feat, sk_feat)
            ranking = torch.argsort(similarities, descending=True)
            ranked_photo_ids = gallery_photo_id[np.array(category_gallery_indices)[ranking.cpu().numpy()]]

            acc_at_1[idx] = float(true_photo_id in ranked_photo_ids[:1])
            acc_at_5[idx] = float(true_photo_id in ranked_photo_ids[:5])

        top1 = acc_at_1.mean()
        top5 = acc_at_5.mean()
        self.log('acc1', top1)
        self.log('acc5', top5)
        self.log('Acc@1', top1)
        self.log('Acc@5', top5)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > top1.item()) else top1.item()
        print('Acc@1: {:.4f}, Acc@5: {:.4f}'.format(top1.item(), top5.item()))
        self.validation_outputs = []
