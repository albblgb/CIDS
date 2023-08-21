mode = 'train' # train or test


# optim Adam
lr = 1e-4
epochs = 3000
weight_decay = 1e-5
weight_step = 500
betas = (0.5, 0.999)
gamma = 0.5

cids_device_ids = '1'
cids_batch_size_train = 8
cids_batch_size_test = 8
cids_beta = 0.5
cids_eps = 0.2

obfuscate_secret_image = False

# dataset
crop_size_train = 256  # size for training
resize_size_test = 512  # size for testing
trainset_path = '/data/gbli/gbData/div2k/train' 
test_secret_image_path = '/data/gbli/gbData/div2k/test'  # survey as the secert dataset when testing
test_cover_image_path = '/data/gbli/gbData/generated_img/garden/0-99'  # the generated dataset, survey as the cover dataset when testing, the number of the images in 'test_cover_image_path' should be same as that of the images in 'test_secret_image_path'

# Saving checkpoints
test_freq = 50
save_freq = 50
save_start_epoch = 1500
model_dir = 'model_zoo'


# Saving processed images
save_processed_img = False
img_save_dir = 'results/images'
testset_name = 'div2k' #  the network-generated images will be saved in 'results/images/*net/testset_name/'
suffix = 'png'


test_cids_path = 'checkpoint_2960.pt'



