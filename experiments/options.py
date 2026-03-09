import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/') 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-5)
parser.add_argument('--prompt_lr', type=float, default=1e-5)
parser.add_argument('--cls_loss_weight', type=float, default=0.5)
parser.add_argument('--triplet_margin', type=float, default=0.3)
parser.add_argument('--patch_shuffle_loss_weight', type=float, default=1.0)
parser.add_argument('--patch_shuffle_grid_size', type=int, default=2)
parser.add_argument('--patch_shuffle_margin', type=float, default=0.3)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--triplet_mode', type=str, default='hard', choices=['category', 'hard'])
parser.add_argument('--match_instance_by_stem', dest='match_instance_by_stem', action='store_true')
parser.add_argument('--no_match_instance_by_stem', dest='match_instance_by_stem', action='store_false')

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)
parser.add_argument('--prompt_init_std', type=float, default=0.02)

parser.set_defaults(match_instance_by_stem=True)

opts = parser.parse_args()
