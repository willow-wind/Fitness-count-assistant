import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import counter  # 动作计数器
import visualizer as vs  # 可视化模块
import pose_embed as pe  # 姿态关键点编码模块
import pose_classify as pc  # 姿态分类器
import result_smooth as rs  # 分类结果平滑


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def video_process(video_path, flag):
    # 指定视频路径和输出名称
    if flag == 1:
        class_name = 'squat_down'

        out_video_path = video_path.replace(".mp4", "_output.mp4")

    elif flag == 2:
        class_name = 'rope_skip'
        out_video_path = video_path.replace(".mp4", "_output.mp4")

    # 打开视频
    video_cap = cv2.VideoCapture(video_path)

    # 获取一些视频参数以生成带有分类的输出视频
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化追踪器、分类器和计数器
    pose_samples_folder = 'squat_csv'

    # 初始化追踪器和嵌入器
    pose_tracker = mp_pose.Pose()
    pose_embedder = pe.FullBodyPoseEmbedder()

    # 初始化分类器
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # 取消注释以验证分类器使用的目标姿态，并找出异常值
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # 初始化EMA平滑
    pose_classification_filter = rs.EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # 初始化计数器
    repetition_counter = counter.RepetitionCounter(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4)

    # 初始化渲染器
    pose_classification_visualizer = vs.PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # 和参数 top_n_by_mean_distance 一致，图形会更加协调
        plot_y_max=10)

    # 在视频中运行分类器

    # 打开视频
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    frame_idx = 0
    output_frame = None
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while True:
            # 获取视频的下一帧
            success, input_frame = video_cap.read()
            if not success:
                break

            # 运行姿态追踪
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            # 绘出姿态预测
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # 获取姿态标志
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # 在当前帧对姿态进行分类
                pose_classification = pose_classifier(pose_landmarks)

                # 平滑分类结果
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # 计算重复次数
                repetitions_count = repetition_counter(pose_classification_filtered)
            else:
                # 没有姿态表明当前帧无分类
                pose_classification = None

                # 向过滤器添加空分类，以保持对未来帧的正确平滑
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # 人不动后停止计数，取最新计数
                repetitions_count = repetition_counter.n_repeats

            # 绘制分类图和重复次数计数器
            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count)

            # 保存输出帧
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

            frame_idx += 1
            pbar.update()

    # 关闭输出视频
    out_video.release()

    # 释放MediaPipe资源
    pose_tracker.close()

    # 展示视频的最后一帧
    if output_frame is not None:
        show_image(output_frame)

