# flag for training or test mode
train = 0  # [1 - True, 0 - False] - Select False when doing only prediction

# root data directory
data_root = 'datasets'

# image directory
img_root = 'just_images_2k'

# experiment type to run classification on
experiment = 'normative'  # ['descriptive', 'normative']

# category to run classification on
category = '1'  # ['1': 'shows_high_skin_exposure',
                # '2': 'contains_text_graphics_images',
                # '3': 'contains_short_shorts_skirt',
                # '0': 'dress code on all categories, logical or of 1,2,3]

# csv file where labels are stored for all categories and experiments - this is assumed to be stored in data_root directory
csv_file = 'labels_dress_classif.csv'

# path to saved model to use in prediction
load_path = ''#'saved_models/1/descriptive/c=1-ex=descriptive-m=resnet50-s=0-ep=0_01-b=64-f=1-t=0.pt'

# Boolean to identify if prediction is across descriptive and normative
cross = 0  # 1 - True if 'load_path' experiment is different from 'experiment' variable

# Boolean to determine if transfer learning is done from descriptive to normative or vice versa
transfer = 0

# path to load model used for transfer learning
transfer_path = ''
#'saved_models/2/normative/c=2-ex=normative-m=resnet50-s=0-ep=0.01-b=64-f=True.pt'

# save test data (use last 1000 images for testing exclusively)
test_only = 1

# Hyperparameter settings
num_images = 2000 # this should be 1000 if we are leaving the last 1000 images for evaluation, otherwise 2000
labels_per_image = 20
input_size = 224  # image dimensions
batch_size = 64
learning_rate = 0.01
ngpu = 4  # number of gpu available, 0 if using cpu
model_name = "resnet50"  # convnet to finutune from
num_epochs = 20#30
num_workers = 10  # number of workers - set to 0 if using windows
feature_extract = 0  # finetune only on last layer of convnet
contention=0.0
split = [0.70, 0.300, 0.001]  # train, val, test split respectively
seed = 0  # random seed used for reproducibility
thresholds = [0, 0.25, 0.5, 0.75, 1] # list of thresholds to test data on
