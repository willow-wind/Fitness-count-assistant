import video_process as vp
import video_capture as vc
import rope_video_process as rv
import rope_videocap as rp

if __name__ == '__main__':
    while True:
        menu = int(input("输入你希望的检测方式：1. 本地视频检测\t2. 摄像头实时检测\t3. 退出\n"))
        if menu == 1:
            flag = int(input("锻炼类型：1. 深蹲\t2. 跳绳\n"))
            video_path = input("请输入视频路径：")
            if flag == 1:
                # tp.trainset_process(flag)
                vp.video_process(video_path, flag)
            if flag == 2:
                # tp.trainset_process(flag)
                # vp.video_process(video_path,flag)
                rv.rope_video_process(video_path)
            continue
        elif menu == 2:
            flag = int(input("锻炼类型：1. 深蹲\t2. 跳绳\n"))
            print("\n按q或esc退出摄像头采集")
            if flag == 1:
                # tp.trainset_process(flag)
                vc.webcam_process()
            if flag == 2:
                rp.rope_video_process()
            continue
        elif menu == 3:
            break
        else:
            print("输错啦，请再输一次~")
            continue
