from lstm_fall_detection import LSTMFallDetection


def main():
    model = LSTMFallDetection()
    model.get_model_summary()
    model.fit_model()


if __name__ == "__main__":
    main()
