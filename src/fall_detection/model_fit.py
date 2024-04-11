from lstm_fall_detection import LSTMFallDetection


def main():
    model = LSTMFallDetection()
    model.getModelSummary()
    model.modelFit()


if __name__ == "__main__":
    main()
