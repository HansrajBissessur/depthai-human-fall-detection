import cv2
import numpy as np

from fall_detection import LSTMFallDetection
from movenet_depthai import MoveNetDepthAI

# Define constants
MODEL_WEIGHTS_PATH = "./fall_detection/model_weights/action2.h5"
FRAME_SEQUENCE_LENGTH = 55


def flatten_keypoints(kps, norm_kps):
    pose = np.concatenate((kps, norm_kps), axis=1)
    print("pose with norms:\n", pose)
    flattened = pose.flatten()
    print("flattened:\n", flattened)
    print("\n")
    return flattened


def main():
    # Initialize fall detection model
    model = LSTMFallDetection()
    model.get_model_summary()
    model.load_model_weights(MODEL_WEIGHTS_PATH)

    # Initialize MoveNetDepthAI
    movenet = MoveNetDepthAI()
    sequence = []  # get 55 frames then pass to detection
    predictions = []

    while True:
        frame, data, _ = movenet.getFrameInference()
        if frame is None:
            break

        # Normalize and resize frame
        if frame is not None:
            normalized = movenet.normalize(data)
            norm_data_frame = movenet.renderNormalized(frame, normalized)
            norm_data_frame = cv2.resize(norm_data_frame, (1920, 1080))

            # Flatten keypoints and add to sequence
            flattened = flatten_keypoints(data, normalized)
            sequence.append(flattened)
            sequence = sequence[-FRAME_SEQUENCE_LENGTH:]

            # Perform prediction if sequence is complete
            if len(sequence) == FRAME_SEQUENCE_LENGTH:
                res = model.predict_model(sequence)

                # Display prediction on frame
                if res < 0.5:
                    current_prediction = f"{res} - fall"
                    cv2.putText(norm_data_frame, current_prediction, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 4, cv2.LINE_AA)
                else:
                    current_prediction = f"{res} - normal"
                    cv2.putText(norm_data_frame, current_prediction, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 200, 0), 4, cv2.LINE_AA)

                predictions.append(np.argmax(res))

            cv2.imshow("Launch Demo", norm_data_frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
