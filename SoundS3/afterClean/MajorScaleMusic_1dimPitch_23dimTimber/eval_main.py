from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

CHECK_POINT_PATH = 'checkpoint_200000.pt'
CHECK_POINT_NUM = int(CHECK_POINT_PATH.split('.')[0].split('_')[-1])
EVAL_NAME = f'eval_{CHECK_POINT_NUM}.txt'

if __name__ == '__main__':
    trainer = BallTrainer(CONFIG, is_train=False)
    trainer.eval_a_checkpoint(CHECK_POINT_NUM, CHECK_POINT_PATH, EVAL_NAME)
