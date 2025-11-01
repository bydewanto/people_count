import cv2
import argparse
from peps_detector.detector import PepsDetector
from peps_detector.utils import save_detections, show_detections, apply_nms


def main():
    parser = argparse.ArgumentParser(
        description="Run object detection using SVM + HOG + LBP."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model", default="models/svm_tuned.sav", help="Path to trained model"
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = cv2.resize(
        image,
        (
            min(1200, image.shape[1]),
            int(image.shape[0] * (min(1200, image.shape[1]) / image.shape[1])),
        ),
    )

    roi = cv2.selectROI(image)
    cropped = image[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ]
    cv2.destroyAllWindows()

    detector = PepsDetector(args.model)
    detections = detector.detect(cropped)

    save_detections(cropped, detections)
    show_detections(cropped, detections, "Raw Detections")

    picks = apply_nms(detections)
    show_detections(
        cropped, [(x, y, 0, w - x, h - y) for (x, y, w, h) in picks], "After NMS"
    )


if __name__ == "__main__":
    main()
    