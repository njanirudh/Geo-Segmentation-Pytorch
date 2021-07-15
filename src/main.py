from src.segmentation_trainer import SegmentationModule

def train_model(dataset:str):
    model_trainer = SegmentationModule(in_channels=12,
                                           out_channels=3,
                                           batch_size=20,
                                           dataset_path=dataset)
    model_trainer.train_model()

if __name__ == "__main__":

    DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    train_model(DATASET_PATH)