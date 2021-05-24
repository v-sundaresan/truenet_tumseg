'''
Triplanar U-Net ensemble network (TrUE-Net) for tumour segmentation

For training, run:
truenet_tumseg train -i <input_directory> -m <model_directory>

For testing, run:
truenet_tumseg evaluate -i <input_directory> -m <model_directory> -o <output_directory>

For leave-one-out validation, run:
truenet_tumseg cross_validate -i <input_directory> -o <output_directory>

for fine-tuning, run:
truenet_tumseg fine_tune -i <input_directory> -m <model_directory> -o <output_directory>
'''