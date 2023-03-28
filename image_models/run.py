import sys
import argparse
import params as p
import ClassifierAll


def program_config(parser):
    parser.add_argument('--data_root', default=p.data_root, type=str)
    parser.add_argument('--img_root', default=p.img_root, type=str)
    parser.add_argument('--experiment', default=p.experiment, type=str)
    parser.add_argument('--load_path', default=p.load_path, type=str)
    parser.add_argument('--category', default=p.category, type=str)
    parser.add_argument('--csv_file', default=p.csv_file, type=str)
    parser.add_argument('--num_images', default=p.num_images, type=int)
    parser.add_argument('--labels_per_image', default=p.labels_per_image, type=int)
    parser.add_argument('--input_size', default=p.input_size, type=int)
    parser.add_argument('--batch_size', default=p.batch_size, type=int)
    parser.add_argument('--learning_rate', default=p.learning_rate, type=float)
    parser.add_argument('--ngpu', default=p.ngpu, type=int)
    parser.add_argument('--model_name', default=p.model_name, type=str)
    parser.add_argument('--num_epochs', default=p.num_epochs, type=int)
    parser.add_argument('--num_workers', default=p.num_workers, type=int)
    parser.add_argument('--feature_extract', default=p.feature_extract, type=int)
    parser.add_argument('--split', type=str, default=p.split)
    parser.add_argument('--seed', default=p.seed, type=int)
    parser.add_argument('--thresholds', nargs='+', default=p.thresholds)
    parser.add_argument('--train', default=p.train, type=int)
    parser.add_argument('--cross', default=p.cross, type=int)
    parser.add_argument('--transfer', default=p.transfer, type=int)
    parser.add_argument('--transfer_path', default=p.transfer_path, type=str)
    parser.add_argument('--test_only', default=p.test_only, type=int)
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--contention', default=p.contention, type=float)
    parser.add_argument('--contention_ref',default='descriptive',type=str)
    parser.add_argument('--train_size_red',default=1, type=float)
    parser.add_argument('--label_noise',default=0,type=float)
    parser.add_argument('--external_dataset',default='data_img_labels.csv',type=str)
    parser.add_argument('--dataset_name',default='dress',type=str)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--weight_decay',default=0.1,type=float)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opts = parser.parse_args()
    if opts.logfile:
        orig_stdout = sys.stdout
        f = open(opts.logfile, 'w')
        sys.stdout = f

    if opts.train==1:
        c = ClassifierAll.Classifier(opts)
        c.run()
    else:
        raise NotImplementedError
    if opts.logfile:
        sys.stdout = orig_stdout
        f.close()
