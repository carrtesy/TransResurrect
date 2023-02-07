import os
import torch

def prepare_arguments(parser):
    '''
    get input arguments
    :return: args
    '''

    # basic config
    parser.add_argument('--exp_id', type=str, default='default', help='exp id')
    parser.add_argument('--model', type=str, required=True, default='Transformer')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--logs', type=str, default='./logs/', help='location of model checkpoints')
    parser.add_argument('--outputs', type=str, default='./outputs/', help='location of model checkpoints')
    parser.add_argument('--plots', type=str, default='./plots/', help='location of model checkpoints')

    # data processing
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--num_channels', type=int, default=7, help='encoder input size')

    # model define
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    args = parser.parse_args()

    print('Args in experiment:')
    configure_exp_id(args)
    print(args)

    args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
    args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
    args.output_path = os.path.join(args.outputs, f"{args.exp_id}")
    args.plot_path = os.path.join(args.plots, f"{args.exp_id}")

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)

    args.home_dir = "."
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

def configure_exp_id(args):
    if args.exp_id == "default":
        id = ""
        id += f"{args.data}_{args.seq_len}_{args.pred_len}_{args.features}"
        id += f"_bs_{args.batch_size}_lr_{args.learning_rate}_wd_{args.weight_decay}_ep{args.train_epochs}"
        id += f"_d_{args.d_model}_ff_{args.d_ff}_el_{args.e_layers}_dl_{args.d_layers}_do_{args.dropout}"
        args.exp_id = id
    return args