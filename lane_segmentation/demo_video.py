import cv2
from lane_segmentation import *


class LaneSegmentationVideo():
    def __init__(self, model='deeplabv3p'):
        super(LaneSegmentationVideo, self).__init__()
        self.lane_seg = LaneSegmentation(model)
    
    def get_mask(self, video_path=None, offset=200, save=False):
        # 选择视频流
        if video_path:
            cap = cv2.VideoCapture(video_path) #从视频中读取图像
            save_video = os.path.splitext(video_path)[0]+'_mask.avi' #单独mask视频
            save_video_combined = os.path.splitext(video_path)[0]+'_mask_combined.avi' #mask和原视频并列
            save_image = os.path.splitext(video_path)[0]+'_mask.jpg'
        else:
            cap = cv2.VideoCapture(0) #调用电脑连接的第0个摄像头(通常为前置）
            save_video = os.path.join(DATA_ROOT, TEST_PATH, 'webcam_mask.avi')
            save_video_combined = os.path.join(DATA_ROOT, TEST_PATH,'webcam_mask_combined.avi') #mask和原视频并
            save_image = os.path.join(DATA_ROOT, TEST_PATH,'webcam_mask.jpg')
        
        # 视频属性
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # avi视频：YUV编码*‘I420’，MPEG-1编码*‘PIMI’，MPEG-4编码*'XVID'
        # flv视频：*‘FLV1’
        fps = cap.get(cv2.CAP_PROP_FPS) 
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_mask = cv2.VideoWriter(save_video, fourcc, fps, (w,h)) #一般为(1280,720)
        out_combined = cv2.VideoWriter(save_video_combined, fourcc, fps, (w,h*2)) #一般为(1280,720)
        
        # 读取视频流
        while cap.isOpened(): #正常打开
            normal, frame = cap.read()
            if not normal: #如果视频播放结束，跳出循环
                break 

            # 预测
            image = cv2.resize(frame[:-offset,:], IMAGE_SIZE) #resize成指定大小
            mask = self.lane_seg.get_mask(image) #(h',w',3)的RGB
            mask = cv2.resize(mask, (w, h-offset), interpolation=cv2.INTER_NEAREST) #恢复剪裁后尺寸
            final_mask = np.zeros((h,w,3), dtype='uint8') #恢复原尺寸
            final_mask[:-offset, ...] = mask
            final_mask = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR) #把mask从RGB转为BGR
            combined = cv2.vconcat([frame, final_mask]) 

            # 输出画面
            cv2.imshow('We are inferring lane mask ... (Press "q" to quick)', final_mask)
            
            # 保存
            if save:
                out_mask.write(final_mask) #逐帧保存 
                out_combined.write(combined)
            if cv2.waitKey(1) & 0xff == ord('q'): #维持窗口，实时检测，按q退出
                break
                
        cv2.imwrite(save_image, final_mask) #保存截图
        # plt.imshow(final_mask[...,::-1]) #把最后一张图片以RGB显示出来
        # plt.axis('off')
        # plt.show()
        cap.release() #断开视频/摄像头
        out_mask.release()
        out_combined.release()
        cv2.destroyAllWindows() #关闭所有窗口
        # cv2.waitKey(1) #只用在Jupyter上，防止kernel奔溃


def show_material(frame, offset=200): #检查需要预测的行车记录仪视频素材
    #参考训练集的画面，先把下半部分有大量车头的区域去掉，再resize成训练图大小
    crop_frame = cv2.resize(frame[:-offset,:], IMAGE_SIZE) #(720, 1280, 3)
    crop_frame = crop_and_resize(crop_frame) #(1710, 3384)
    images = [frame, crop_frame]
    plt.figure(figsize=(10,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(images[i][...,::-1])
        plt.title(images[i].shape)
        plt.axis('off')
    # plt.show()


if __name__=='__main__':
    # cap = cv2.VideoCapture('dataset/Test/road_video_compressed2.mp4') #从视频中读取图像
    # for i in range(100): #遍历到第100帧画面
    #     normal, frame = cap.read() #(720, 1280, 3)
    # show_material(frame)

    seg = LaneSegmentationVideo()
    seg.get_mask('dataset/Test/road_video_compressed2.mp4', save=True)