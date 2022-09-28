from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

if __name__ == '__main__':
    trainer = BallTrainer(CONFIG)
    if is_need_train(CONFIG):
        trainer.train()
