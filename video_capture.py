from matplotlib import pyplot as plt
import cv2
import numpy as np
# import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import pose_embed as pe  # 姿态关键点编码模块
import pose_classify as pc  # 姿态分类器
import result_smooth as rs  # 分类结果平滑
import counter  # 动作计数器
import visualizer as vs  # 可视化模块


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def webcam_process():
    class_name = 'squat_down'
    out_video_path = 'squat-sample-out.mp4'
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    video_cap = cv2.VideoCapture(0)
    # video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = 24
    video_width = 640
    video_height = 480

    pose_samples_folder = 'squat_csv'
    pose_tracker = mp_pose.Pose()
    pose_embedder = pe.FullBodyPoseEmbedder()
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    pose_classification_filter = rs.EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    repetition_counter = counter.RepetitionCounter(
        class_name=class_name,
        enter_threshold=5,
        exit_threshold=4)

    pose_classification_visualizer = vs.PoseClassificationVisualizer(
        class_name=class_name,
        plot_y_max=10)

    # 运行分类器

    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
    output_frame = None
    while video_cap.isOpened():
        success, input_frame = video_cap.read()
        if not success:
            break

        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            pose_classification = pose_classifier(pose_landmarks)

            pose_classification_filtered = pose_classification_filter(pose_classification)

            repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            pose_classification = None
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None
            repetitions_count = repetition_counter.n_repeats

        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # 实时输出检测画面
        cv2.imshow('video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
        # 按键盘的q或者esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    out_video.release()
    video_cap.release()
    cv2.destroyAllWindows()

    pose_tracker.close()

    if output_frame is not None:
        show_image(output_frame)

