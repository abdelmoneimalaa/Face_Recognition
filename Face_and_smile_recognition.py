import os
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image 

def initialize_models():
    mtcnn = MTCNN(device="cpu", margin=20)
    model = InceptionResnetV1(pretrained='vggface2').eval().to("cpu")
    return mtcnn, model

def get_image_files(image_path):
    return os.listdir(image_path)

def process_frame(frame, mtcnn, model):
    rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_cropped = mtcnn(rgb_frame)
    if img_cropped is not None:
        frame_embedding = model(img_cropped.unsqueeze(0).to("cpu")).detach().numpy()
        return frame_embedding
    else:
        return None

def compare_embeddings(frame_embedding, image_files, image_path, mtcnn, model,threshold):
    for image_file in image_files:
        img = Image.open(image_path / image_file)
        img_cropped = mtcnn(img)
        img_embedding = model(img_cropped.unsqueeze(0).to("cpu")).detach().numpy()
        dist = np.linalg.norm(frame_embedding - img_embedding)
        if dist < threshold:
            return image_file
    return None

def main():
    threshold = 1
    cap = cv2.VideoCapture(0)
    trained_smile_data = cv2.CascadeClassifier('Smile/smile_cascade.xml')
    mtcnn, model = initialize_models()
    data_path = Path("pic/")
    image_path = data_path / "Employee"
    image_files = get_image_files(image_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

        frame_embedding = process_frame(frame, mtcnn, model)
        if frame_embedding is not None:
            matching_image_file = compare_embeddings(frame_embedding, image_files, image_path, mtcnn, model,1)
            if matching_image_file is not None:
                print(f"Match found with {matching_image_file[:-4]}")
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                smile_coords = trained_smile_data.detectMultiScale(gray_frame, scaleFactor=1.7, minNeighbors=20)
                if len(smile_coords) != 0:
                    print(f"{matching_image_file[:-4]} is also smiling")
                else:
                    print(f"{matching_image_file[:-4]} is NOT smiling")
            else:
                print("No match found")
        else:
            print("No face detected")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
