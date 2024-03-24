import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_images', type=str, default='T:/dataset_exp/256/img/',
                        help='directory to images')
    parser.add_argument('--path_to_labels', type=str, default='T:/dataset_exp/256/label-h/',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str,default="F:/IEL2/dataset/csv/",
                        help='directory to csv_folder')
    parser.add_argument('--experiment_dir_name', type=str, default='F:/IEL2/weights/',
                        help='directory to project weights saved')

    parser.add_argument('--path_to_model_weight', type=str,
                        default='F:/IEL2/weights/UNet3+_leakyrelu1_l1loss_2021_12_11_17_19_epoch_3_tloss_0.040703118362985395_vloss0.3013541749792128.pth',
                        help='directory to pretrain model')
    #model_weight
    parser.add_argument('--init_lr', type=int, default=0.0001, help='learning_rate')
    parser.add_argument('--num_epoch', type=int, default=25, help='epoch num for train')
    parser.add_argument('--batch_size', type=int, default=16, help='how many pictures to process one time')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')

    args = parser.parse_args()
    return args