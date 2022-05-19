import streamlit as st
import os
import torch
from PIL import Image
import cv2
from yolov5.detect import run 

#-------------------------------------------- 3. Layout -------------------------------------------------------------
st.set_page_config(
    page_title = 'HackBlue R&D Team',
    page_icon = 'https://pisces.bbystatic.com/image2/BestBuy_US/Gallery/Favicon-152-72229.png',
    layout = 'wide'
)
st.markdown(
     """
     <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-image: linear-gradient(#0A4ABF,#0A4ABF);
        color: white
     }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        background-image: linear-gradient(#0A4ABF,#0A4ABF);
        color: white
     }

    [data-baseweb="radio"] > div:first-child {
        background-image: linear-gradient(#FFED31,#FFED31);
        color: #FFED31
        text-color: #FFED31
     }

        </style>
        """,
        unsafe_allow_html=True)


layout = st.sidebar.columns([2, 3])

#st.sidebar.title('HackBlue R&D Team') # make a markdown and make it yellow if background is blue
#st.sidebar.markdown(f'<p style="font-size: 30px; color: #FFED31;"><b>HackBlue R&D Team</b></p>', unsafe_allow_html=True)

st.sidebar.image('logow.png', width=170)
#st.sidebar.markdown('<div><img src="https://pisces.bbystatic.com/image2/BestBuy_US/Gallery/Favicon-152-72229.png" style="display: flex;align-items: center;justify-content: center;"></div>', unsafe_allow_html=True)
#-------------------------------------- 4. User selection field -----------------------------------------------------

model = torch.load('best.pt')

choice = st.sidebar.radio("Select", ('Image', 'Video'))

file = st.sidebar.file_uploader("Choose a file", type=["png","jpg","jpeg","mp4","avi"])

st.markdown(f'<p style="font-size: 30px; color: #0A4ABF; text-align: center;"><b>Team10-Data Science R&D</b></p>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html = True)

extensions_img = ['jpg','png','jpeg']
extensions_vid = ['mp4','avi']

if file != None:
    
    with open(os.path.join("input/",file.name),"wb") as f:
        f.write((file).getbuffer())

    f_name = file.name
    f_name = f_name.split('.')[1]

    if (choice == 'Image' and  f_name in extensions_img) or (choice == 'Video' and  f_name in extensions_vid):       
        dict = run(
        weights='best.pt',  # model.pt path(s)
        source=f"input/{file.name}",  # file/dir/URL/glob, 0 for webcam
        data='data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='hack_blue',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    )

        path = f'runs/detect/hack_blue/{file.name}'

    else:

        st.markdown(f'<p style="font-size: 30px; color: #FF0000;"><b>Error. You have selected the wrong file type.</b></p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 30px; color: #0A4ABF;"><b>If you want to predict an image use a file with the extensions: jpg, jpeg, png.</b></p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 30px; color: #0A4ABF;"><b>If you want to predict a video use a file with the extensions: mp4, avi.</b></p>', unsafe_allow_html=True)

    
    
    if choice == 'Image':

        if f_name in extensions_img:

            row1, row2 = st.columns((3, 3))
                
            with row1:
                st.markdown(f'<p style="font-size: 25px; color: #0A4ABF;"><b>Original Image</b></p>', unsafe_allow_html=True)
                st.image(file)

            with row2:
                st.markdown(f'<p style="font-size: 25px; color: #0A4ABF;"><b>Processed Image</b></p>', unsafe_allow_html=True)
                st.image(path)

            st.markdown("<hr/>", unsafe_allow_html = True)
            
            row3, row4, row5 = st.columns((2, 2, 2))
            
            with row3:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Image count: </b>{dict["avg_total"]}</p>', unsafe_allow_html=True) #talk about total detections
            with row4:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Count on the left: </b>{dict["avg_left"]}</p>', unsafe_allow_html=True)
            with row5:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Count on the right: </b>{dict["avg_right"]}</p>', unsafe_allow_html=True)
        
    else:
        if f_name in extensions_vid:
            row1, row2 = st.columns((3, 3))
            
            video = cv2.VideoCapture(f"input/{file.name}")

            # the frame rate or frames per second
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))

            # the total number of frames
            total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # the duration in seconds
            duration = total_num_frames // frame_rate
            #duration = video.get(cv2.CAP_PROP_POS_MSEC)
            

            with row1:
                st.markdown(f'<p style="font-size: 25px; color: #0A4ABF;"><b>Original Video</b></p>', unsafe_allow_html=True)
                st.video(file)

            with row2:
                st.markdown(f'<p style="font-size: 25px; color: #0A4ABF;"><b>Processed Video</b></p>', unsafe_allow_html=True)

                video_file = open(path, 'rb')
                video_s = video_file.read()
                st.video(video_s)

            st.markdown("<hr/>", unsafe_allow_html = True)
            
            st.markdown(f'<p style="text-align: center; font-size: 25px; color: #0A4ABF;"><b>Overall Statistics</b></p>', unsafe_allow_html=True)
            row3, row4, row5 = st.columns((2, 2, 2))

            with row3:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Average count: </b>{dict["avg_total"]}</p>', unsafe_allow_html=True) #talk about total detections
            with row4:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Average count on the left: </b>{dict["avg_left"]}</p>', unsafe_allow_html=True)
            with row5:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Average count on the right: </b>{dict["avg_right"]}</p>', unsafe_allow_html=True)
            
            st.markdown("<hr/>", unsafe_allow_html = True)
            st.markdown(f'<p style="text-align: center; font-size: 25px; color: #0A4ABF;"><b>Time Statistics</b></p>', unsafe_allow_html=True)
            row6, row7, row8 = st.columns((2, 2, 2))
            
            with row6:
                st.markdown(f'<p style="text-align: center; font-size: 20px; color: #0A4ABF;"><b>Overall Video</b></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Maximun: </b>{(max(dict["max_min_frames"]))[0]} <b> detections at time stamp</b> {((max(dict["max_min_frames"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Minimum: </b>{(min(dict["max_min_frames"]))[0]} <b> detections at time stamp</b> {((min(dict["max_min_frames"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
            with row7:
                st.markdown(f'<p style="text-align: center; font-size: 20px; color: #0A4ABF;"><b>Left Area</b></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Maximun: </b>{(max(dict["max_min_left"]))[0]} <b> detections at time stamp</b> {((max(dict["max_min_left"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Minimum: </b>{(min(dict["max_min_left"]))[0]} <b> detections at time stamp</b> {((min(dict["max_min_left"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
            with row8:
                st.markdown(f'<p style="text-align: center; font-size: 20px; color: #0A4ABF;"><b>Right Area</b></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Maximun: </b>{(max(dict["max_min_right"]))[0]} <b> detections at time stamp</b> {((max(dict["max_min_right"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Minimum: </b>{(min(dict["max_min_right"]))[0]} <b> detections at time stamp</b> {((min(dict["max_min_right"]))[1])//frame_rate}</p>', unsafe_allow_html=True)
    # 23 DETCTCIONTS AT TIME STAMP 5
            st.markdown("<hr/>", unsafe_allow_html = True)
            st.markdown(f'<p style="text-align: center; font-size: 25px; color: #0A4ABF;"><b>Video Statistics</b></p>', unsafe_allow_html=True)
            row9, row10, row11 = st.columns((2, 2, 2))
            
            with row9:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Video duration: </b>{duration} seconds</p>', unsafe_allow_html=True)
            with row10:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Total frames: </b>{total_num_frames}</p>', unsafe_allow_html=True)
            with row11:
                st.markdown(f'<p style="font-size: 20px; color: #1C252C;"><b>Frames per second: </b>{frame_rate}</p>', unsafe_allow_html=True)
                
            
            


