import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts

def is_legacy_prompt_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    return 'sk_prompt' in state_dict or 'img_prompt' in state_dict

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.workers,
        shuffle=True,
        drop_last=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.workers,
        shuffle=False,
        drop_last=False)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='acc1',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{acc1:.4f}",
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    elif is_legacy_prompt_checkpoint(ckpt_path):
        print('found checkpoint with separate sketch/photo prompts; starting fresh for common-prompt architecture')
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(accelerator=accelerator,
        devices=1,
        min_epochs=1, max_epochs=60,
        benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )

    model_categories = train_dataset.all_categories
    if ckpt_path is None:
        model = Model(categories=model_categories)
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model.load_from_checkpoint(ckpt_path, categories=model_categories)

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
