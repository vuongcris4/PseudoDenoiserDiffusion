# Dataset config for D3PM pseudo-label denoising.
# Uses OEMCISCRCrossEntropyDataset with split files (train.txt, val.txt, test.txt).

dataset_type = 'OEMCISCRCrossEntropyDataset'
data_root = 'data/OEM_v2_aDanh'
num_classes = 7

img_size = 512

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        img_size=img_size,
        augment=True,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        img_size=img_size,
        augment=False,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        img_size=img_size,
        augment=False,
        num_classes=num_classes))
