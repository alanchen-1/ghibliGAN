from .options import Options

class TrainOptions(Options):

    def initialize(self, parser):
        parser.add_argument('--save_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--start_epoch', type=int, default=1, help='the starting epoch')
        parser.add_argument('--phase', type=str, default='train', help='which phase the model is in')

        parser.add_argument('--warmup_epochs', type=int, default=100, help='number of epochs with constant lr. this + decay_epochs = total epochs')
        parser.add_argument('--decay_epochs', type=int, default=100, help='number of epochs with decaying lr. this + warmup_epochs = total epochs')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 parameter for adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--loss_type', type=str, default='mse', help='loss criteria to use')
        parser.add_argument('--buffer_size', type=int, default=50, help='size of image buffer to use, original paper uses 50 (default)')

        parser.add_argument('--lambda_scaling', type=float, default=0.5, help='scaling multiplier to use in loss calculation')
        parser.add_argument('--lambda_X', type=float, default=10.0, help='multiplier on cycle loss in domain X (i.e. on the cycle XYX)')
        parser.add_argument('--lambda_Y', type=float, default=10.0, help='multiplier on cycle loss in domain Y (i.e. on the cycle YXY)')

        self.to_train = True
        return parser